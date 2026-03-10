"""Paper Trading Service — bridges pipeline decisions to Alpaca paper trading.

In paper mode, the full pipeline runs (Cognition → Regime → Decision →
Execution validation), and orders are submitted to Alpaca's paper trading
endpoint. No real money is exchanged, but execution is realistic.

Like ShadowExecutionService, signals are also recorded as ShadowSignals
for dual tracking and later accuracy analysis.

Usage:
    paper_svc = PaperTradingService(
        broker_client=alpaca_client,
        order_manager=order_manager,
        portfolio_tracker=portfolio_tracker,
        shadow_signal_store=shadow_signal_store,  # optional
    )
    execution_summary = await paper_svc.execute_signals(run_result)
    await paper_svc.reconcile()
    await paper_svc.emergency_halt()
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

import structlog

from providence.exceptions import ExternalAPIError
from providence.infra.alpaca_client import AlpacaClient
from providence.orchestration.models import PipelineRun, StageStatus
from providence.portfolio.order_manager import OrderManager, OrderStatus, ManagedOrder
from providence.portfolio.tracker import PortfolioTracker
from providence.schemas.enums import Action, Direction, SystemMode
from providence.services.shadow_execution import ShadowExecutionService, ShadowSignalStore

logger = structlog.get_logger()


class PaperTradingService:
    """Manages paper trading execution with dual signal tracking.

    Flow:
    1. execute_signals() — extract validated proposals and submit to Alpaca
    2. poll_order_status() — check fills with broker
    3. reconcile() — sync portfolio state with broker
    4. emergency_halt() — cancel all orders + optionally flatten positions
    5. Optional: record_signals() — also record as ShadowSignals for analysis

    Thread-safe. All operations are async.
    """

    POLL_INTERVAL_SECONDS = 2.0
    MAX_POLL_ATTEMPTS = 30  # 60 seconds max

    def __init__(
        self,
        broker: AlpacaClient,
        order_manager: OrderManager,
        portfolio: PortfolioTracker,
        shadow_signal_store: ShadowSignalStore | None = None,
    ):
        self._broker = broker
        self._orders = order_manager
        self._portfolio = portfolio
        self._shadow_signal_store = shadow_signal_store
        self._shadow_svc: ShadowExecutionService | None = None
        if shadow_signal_store is not None:
            self._shadow_svc = ShadowExecutionService(shadow_signal_store)

    async def execute_signals(
        self,
        run_result: PipelineRun,
    ) -> dict[str, Any]:
        """Execute signals from a pipeline run to paper trading.

        Extracts EXEC-VALIDATE and EXEC-ROUTER outputs, submits orders
        to Alpaca paper trading, polls for fills, and optionally records
        signals as ShadowSignals for dual tracking.

        Args:
            run_result: Complete PipelineRun from main loop

        Returns:
            Execution summary with keys:
                - submitted: number of orders submitted
                - filled: number of orders that filled
                - rejected: number rejected by broker
                - cancelled: number cancelled
                - halted: whether system halt was triggered
                - signal_summary: ShadowRunSummary if shadow recording enabled
                - orders: list of ManagedOrder summaries
                - error: error message if any
        """
        log = logger.bind(
            run_id=str(run_result.run_id),
            mode="PAPER",
            total_stages=len(run_result.stage_results),
        )
        log.info("Executing paper trading signals")

        # Extract EXEC-VALIDATE, EXEC-ROUTER, and EXEC-GUARDIAN outputs
        validated_proposal = None
        routing_plan = None
        guardian_verdict = None
        regime_state = None

        for sr in run_result.stage_results:
            if sr.agent_id == "EXEC-VALIDATE" and sr.status == StageStatus.SUCCEEDED:
                validated_proposal = sr.output
            elif sr.agent_id == "EXEC-ROUTER" and sr.status == StageStatus.SUCCEEDED:
                routing_plan = sr.output
            elif sr.agent_id == "EXEC-GUARDIAN" and sr.status == StageStatus.SUCCEEDED:
                guardian_verdict = sr.output
            elif sr.agent_id == "REGIME-MISMATCH" and sr.status == StageStatus.SUCCEEDED:
                regime_state = sr.output

        if not validated_proposal or not isinstance(validated_proposal, dict):
            log.warning("No EXEC-VALIDATE output found for paper execution")
            return {
                "submitted": 0,
                "filled": 0,
                "rejected": 0,
                "cancelled": 0,
                "halted": False,
                "error": "No validated proposal",
                "orders": [],
            }

        # Check guardian verdict for system halt
        system_halt = False
        halt_reason = ""
        if guardian_verdict and isinstance(guardian_verdict, dict):
            system_halt = guardian_verdict.get("system_halt", False)
            halt_reason = guardian_verdict.get("halt_reason", "")

        if system_halt:
            log.warning(
                "System halt triggered by guardian",
                reason=halt_reason,
            )
            halt_result = await self.emergency_halt()
            return {
                "submitted": 0,
                "filled": 0,
                "rejected": 0,
                "cancelled": halt_result.get("cancelled_orders", 0),
                "halted": True,
                "halt_reason": halt_reason,
                "orders": [],
            }

        # Get portfolio equity for position sizing
        try:
            account = await self._broker.get_account()
            portfolio_equity = Decimal(str(account.get("portfolio_value", 0)))
        except ExternalAPIError as e:
            log.error(
                "Failed to get account for portfolio sizing",
                error=str(e),
            )
            return {
                "submitted": 0,
                "filled": 0,
                "rejected": 0,
                "cancelled": 0,
                "halted": False,
                "error": f"Failed to fetch account: {str(e)}",
                "orders": [],
            }

        # Create ManagedOrders from RoutedOrders (from EXEC-ROUTER)
        managed_orders = []
        if routing_plan and isinstance(routing_plan, dict):
            routed_orders = routing_plan.get("orders", [])
            for routed_order in routed_orders:
                try:
                    managed_order = self._orders.create_from_routed_order(
                        routed_order,
                        portfolio_equity=portfolio_equity,
                    )
                    managed_orders.append(managed_order)
                except Exception as e:
                    log.error(
                        "Failed to create managed order from routed order",
                        ticker=routed_order.get("ticker"),
                        error=str(e),
                    )

            log.info(
                "Created managed orders from routing plan",
                count=len(managed_orders),
            )
        else:
            log.warning("No routing plan found for paper execution")

        # Submit each order to broker
        submitted_count = 0
        rejected_count = 0

        for managed_order in managed_orders:
            success = await self._submit_order(managed_order)
            if success:
                submitted_count += 1
            else:
                rejected_count += 1

        log.info(
            "Order submission round complete",
            submitted=submitted_count,
            rejected=rejected_count,
        )

        # Poll for fills
        filled_count = await self.poll_order_status()

        # Optionally record signals as ShadowSignals for dual tracking
        signal_summary = None
        if self._shadow_svc is not None:
            signal_summary = self._shadow_svc.record_signals(
                run_id=run_result.run_id,
                validated_proposal=validated_proposal,
                regime_state=regime_state if isinstance(regime_state, dict) else {},
                metadata=run_result.metadata,
            )
            log.info(
                "Paper signals recorded as ShadowSignals for tracking",
                total_signals=signal_summary.total_signals,
                approved=signal_summary.approved_signals,
            )

        # Prepare response
        return {
            "submitted": submitted_count,
            "filled": filled_count,
            "rejected": rejected_count,
            "cancelled": 0,
            "halted": False,
            "signal_summary": signal_summary.model_dump() if signal_summary else None,
            "orders": [o.to_dict() for o in managed_orders],
        }

    async def _submit_order(self, order: ManagedOrder) -> bool:
        """Submit a single ManagedOrder to Alpaca paper trading.

        Args:
            order: ManagedOrder to submit

        Returns:
            True if submission succeeded, False otherwise
        """
        try:
            logger.debug(
                "Submitting paper order to Alpaca",
                order_id=str(order.order_id),
                ticker=order.ticker,
                side=order.side.value,
                type=order.order_type,
            )

            # Submit to Alpaca paper trading
            broker_response = await self._broker.submit_order(
                ticker=order.ticker,
                qty=int(order.qty) if order.qty else None,
                notional=float(order.notional) if order.notional else None,
                side=order.side.value.lower(),
                type=order.order_type,
                time_in_force=order.time_in_force,
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                client_order_id=order.client_order_id,
            )

            broker_order_id = broker_response.get("id")
            if not broker_order_id:
                logger.error(
                    "Broker response missing order ID",
                    order_id=str(order.order_id),
                    response=broker_response,
                )
                self._orders.mark_failed(order.order_id, "Broker response missing order ID")
                return False

            # Mark order as submitted
            self._orders.mark_submitted(order.order_id, broker_order_id)

            logger.info(
                "Paper order submitted successfully",
                order_id=str(order.order_id),
                broker_order_id=broker_order_id,
                ticker=order.ticker,
            )
            return True

        except ExternalAPIError as e:
            logger.error(
                "Broker API error during paper order submission",
                order_id=str(order.order_id),
                ticker=order.ticker,
                status_code=e.status_code,
                error=str(e),
            )

            # Determine if retryable
            if e.status_code == 403:
                self._orders.mark_failed(order.order_id, f"Insufficient funds: {e.message}")
            elif e.status_code == 422:
                self._orders.mark_failed(order.order_id, f"Validation error: {e.message}")
            else:
                self._orders.mark_failed(order.order_id, f"Broker error: {str(e)}")

            return False

        except Exception as e:
            logger.error(
                "Unexpected error during paper order submission",
                order_id=str(order.order_id),
                error=str(e),
            )
            self._orders.mark_failed(order.order_id, f"Unexpected error: {str(e)}")
            return False

    async def poll_order_status(self, max_attempts: int | None = None) -> int:
        """Poll Alpaca for status updates on all active orders.

        Polls for fills and updates order manager state accordingly.
        Records fills in portfolio tracker.

        Args:
            max_attempts: Maximum number of polling attempts.

        Returns:
            Number of orders that reached terminal state
        """
        if max_attempts is None:
            max_attempts = self.MAX_POLL_ATTEMPTS

        terminal_count = 0
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            active_orders = self._orders.get_active_orders()

            if not active_orders:
                logger.debug("No active orders to poll", attempt=attempt)
                break

            logger.debug(
                "Polling paper order status",
                active_count=len(active_orders),
                attempt=attempt,
            )

            for managed_order in active_orders:
                try:
                    # Fetch broker status
                    if managed_order.broker_order_id:
                        broker_order = await self._broker.get_order(
                            managed_order.broker_order_id
                        )
                    else:
                        broker_order = await self._broker.get_order_by_client_id(
                            managed_order.client_order_id
                        )

                    if not broker_order:
                        logger.warning(
                            "Broker order not found",
                            order_id=str(managed_order.order_id),
                            broker_order_id=managed_order.broker_order_id,
                        )
                        continue

                    # Update order state
                    old_status = managed_order.status
                    self._orders.update_from_broker(managed_order.order_id, broker_order)

                    # If newly filled, record in portfolio tracker
                    new_status = managed_order.status
                    if old_status != new_status and new_status == OrderStatus.FILLED:
                        try:
                            await self._portfolio.record_fill(
                                ticker=managed_order.ticker,
                                quantity=managed_order.filled_qty,
                                avg_price=managed_order.filled_avg_price,
                                side=managed_order.side,
                            )
                            logger.info(
                                "Fill recorded in portfolio",
                                order_id=str(managed_order.order_id),
                                ticker=managed_order.ticker,
                                qty=str(managed_order.filled_qty),
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to record fill in portfolio",
                                order_id=str(managed_order.order_id),
                                error=str(e),
                            )

                    if managed_order.is_terminal:
                        terminal_count += 1

                except ExternalAPIError as e:
                    logger.error(
                        "Error polling paper order status",
                        order_id=str(managed_order.order_id),
                        error=str(e),
                    )
                except Exception as e:
                    logger.error(
                        "Unexpected error polling paper order",
                        order_id=str(managed_order.order_id),
                        error=str(e),
                    )

            # Check if all orders are terminal
            remaining_active = self._orders.get_active_orders()
            if not remaining_active:
                logger.info("All paper orders reached terminal state")
                break

            # Wait before next poll
            if attempt < max_attempts:
                await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

        logger.info(
            "Paper order status polling complete",
            attempts=attempt,
            terminal_count=terminal_count,
        )
        return terminal_count

    async def reconcile(self) -> dict[str, Any]:
        """Full reconciliation with Alpaca paper trading state.

        Syncs positions and account data. Detects orphaned orders and
        mismatched fills.

        Returns:
            Reconciliation report with keys:
                - account_synced: bool
                - positions_synced: list[str] (tickers)
                - discrepancies: list[dict]
                - equity: current portfolio equity
                - cash: current cash
        """
        logger.info("Starting full reconciliation with Alpaca paper trading")

        discrepancies = []
        account_synced = False
        positions_synced = []

        try:
            # Get account data and positions from broker
            account = await self._broker.get_account()
            broker_positions = await self._broker.list_positions()
            equity = Decimal(str(account.get("portfolio_value", 0)))
            cash = Decimal(str(account.get("cash", 0)))

            # Sync portfolio tracker
            self._portfolio.sync_from_broker(account, broker_positions)
            account_synced = True

            logger.debug(
                "Account synced",
                equity=str(equity),
                cash=str(cash),
            )

        except ExternalAPIError as e:
            logger.error(
                "Failed to fetch account for paper reconciliation",
                error=str(e),
            )
            return {
                "account_synced": False,
                "positions_synced": [],
                "discrepancies": [],
                "error": str(e),
            }

        try:
            # Detect orphaned positions
            broker_tickers = {pos.get("symbol") for pos in broker_positions}

            # Get portfolio positions
            portfolio_snapshot = self._portfolio.snapshot()
            portfolio_tickers = set(portfolio_snapshot.positions.keys())

            # Check for orphaned positions
            orphaned = broker_tickers - portfolio_tickers
            if orphaned:
                logger.warning(
                    "Orphaned positions found on Alpaca paper trading",
                    orphaned=list(orphaned),
                )
                for ticker in orphaned:
                    discrepancies.append({
                        "type": "orphaned_position",
                        "ticker": ticker,
                        "description": f"Position {ticker} on broker but not in portfolio tracker",
                    })

            positions_synced = list(broker_tickers)

        except ExternalAPIError as e:
            logger.error(
                "Failed to list positions for paper reconciliation",
                error=str(e),
            )

        logger.info(
            "Paper reconciliation complete",
            account_synced=account_synced,
            positions_synced_count=len(positions_synced),
            discrepancies_count=len(discrepancies),
        )

        return {
            "account_synced": account_synced,
            "positions_synced": positions_synced,
            "discrepancies": discrepancies,
            "equity": str(equity) if account_synced else None,
            "cash": str(cash) if account_synced else None,
        }

    async def emergency_halt(self, flatten_positions: bool = False) -> dict[str, Any]:
        """Cancel all orders. Optionally flatten all positions.

        Args:
            flatten_positions: If True, also close all open positions.

        Returns:
            Halt report with keys:
                - cancelled_orders: number of orders cancelled
                - closed_positions: list[str] (tickers of closed positions)
                - error: error message if any
        """
        logger.warning(
            "PAPER TRADING EMERGENCY HALT initiated",
            flatten_positions=flatten_positions,
        )

        cancelled_count = 0
        closed_positions = []
        error = None

        try:
            # Cancel all open orders
            await self._broker.cancel_all_orders()
            active_orders = self._orders.get_active_orders()
            cancelled_count = len(active_orders)

            for order in active_orders:
                try:
                    order.transition_to(
                        OrderStatus.CANCELLED,
                        reason="Paper trading emergency halt triggered",
                    )
                except ValueError:
                    # Force cancel if transition is invalid
                    order.status = OrderStatus.CANCELLED

            logger.info(
                "All paper orders cancelled",
                count=cancelled_count,
            )

        except ExternalAPIError as e:
            error = f"Failed to cancel orders: {str(e)}"
            logger.error(error)

        # Optionally flatten all positions
        if flatten_positions:
            try:
                close_orders = await self._broker.close_all_positions()
                closed_positions = [
                    order.get("symbol", "UNKNOWN") for order in close_orders
                ]
                logger.info(
                    "All paper positions closed",
                    count=len(closed_positions),
                )
            except ExternalAPIError as e:
                error = f"Failed to close positions: {str(e)}"
                logger.error(error)

        logger.warning(
            "Paper trading emergency halt complete",
            cancelled_orders=cancelled_count,
            closed_positions_count=len(closed_positions),
            error=error,
        )

        return {
            "cancelled_orders": cancelled_count,
            "closed_positions": closed_positions,
            "error": error,
        }

    async def retry_failed_orders(self) -> int:
        """Retry orders that failed but haven't exceeded max retries.

        Returns:
            Number of orders retried
        """
        retryable_orders = self._orders.get_retryable_orders()

        if not retryable_orders:
            logger.debug("No retryable paper orders found")
            return 0

        logger.info(
            "Retrying failed paper orders",
            count=len(retryable_orders),
        )

        retry_count = 0
        for order in retryable_orders:
            self._orders.increment_retry_count(order.order_id)

            try:
                if order.status in (OrderStatus.FAILED, OrderStatus.REJECTED):
                    order.status = OrderStatus.PENDING
                    success = await self._submit_order(order)
                    if success:
                        retry_count += 1
            except Exception as e:
                logger.error(
                    "Error retrying paper order",
                    order_id=str(order.order_id),
                    error=str(e),
                )

        logger.info(
            "Paper order retry round complete",
            retried=retry_count,
        )
        return retry_count

    @property
    def pending_orders_count(self) -> int:
        """Number of paper orders awaiting submission."""
        return len(self._orders.get_pending_orders())

    @property
    def active_orders_count(self) -> int:
        """Number of active (non-terminal) paper orders."""
        return len(self._orders.get_active_orders())

    @property
    def order_stats(self) -> dict[str, int]:
        """Get paper order stats from order manager."""
        return self._orders.stats
