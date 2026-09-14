/**
 * web/static/js/dashboard.js
 * Real-time SSE listener, polling fallback, and operator action dispatchers.
 */

// =============================================================================
// State Management
// =============================================================================
const AppState = {
    sseConnected: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    eventSource: null,
    pollingInterval: null,
    inFlightAction: false,
    feed_source: 'live_alpaca',
    alert_banner_active: false,
    current_regime: null,
    feedbackTimer: null,
};

// =============================================================================
// Formatting Utilities
// =============================================================================
function formatCurrency(val) {
    if (typeof val !== 'number' || isNaN(val)) val = 0.0;
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(val);
}

function formatPercent(val) {
    if (typeof val !== 'number' || isNaN(val)) val = 0.0;
    const sign = val > 0 ? '+' : '';
    return `${sign}${(val * 100).toFixed(2)}%`;
}

// =============================================================================
// DOM Update Engine
// =============================================================================
function updateRegimeDesc(regime) {
    const regimeDesc = document.getElementById('regime-desc');
    if (!regimeDesc || !regime) return;

    let descText = '';
    if (regime === 'BULL_MOMENTUM') {
        descText = 'Bull Trend & Momentum Risk On';
    } else if (regime === 'REBOUND_BOUNCE') {
        descText = 'Rebound Bounce & Recovery Trend';
    } else if (regime === 'HIGH_VOLATILITY') {
        descText = 'High Volatility & Defensives';
    } else if (regime === 'CRASH_DEFENSE') {
        descText = 'Downside Protection & Cash/Bonds';
    } else {
        descText = regime.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
    }
    regimeDesc.textContent = descText;
}

function updateStatusUI(data) {
    if (!data) return;

    // 1. Status Badge
    const badge = document.getElementById('status-badge');
    const state = data.state || (data.status && data.status.state);
    if (badge && state) {
        badge.textContent = state;
        badge.className = 'px-2.5 py-1 text-xs font-bold rounded-full border transition-colors duration-200 ';
        if (state === 'RUNNING') {
            badge.className += 'bg-emerald-50 text-emerald-700 border-emerald-200';
        } else if (state === 'PAUSED') {
            badge.className += 'bg-amber-50 text-amber-700 border-amber-200';
        } else {
            badge.className += 'bg-rose-50 text-rose-700 border-rose-200';
        }
    }

    // 2. Alert Banner (Feature 13)
    if (data.feed_source !== undefined) {
        AppState.feed_source = data.feed_source;
    }
    if (data.alert_banner_active !== undefined) {
        AppState.alert_banner_active = data.alert_banner_active;
    }

    const banner = document.getElementById('alert-banner');
    if (banner) {
        const isFallback = AppState.feed_source === 'synthetic_fallback' || AppState.alert_banner_active === true;
        if (isFallback) {
            banner.classList.remove('hidden');
        } else {
            banner.classList.add('hidden');
        }
    }

    // 3. Regime Badge
    const regime = data.regime || data.current_regime;
    if (regime) {
        AppState.current_regime = regime;
        const regimeBadge = document.getElementById('regime-badge');
        if (regimeBadge) {
            regimeBadge.textContent = regime;
        }
        updateRegimeDesc(regime);
    }

    // 4. Button States (dim active state)
    const btnPause = document.getElementById('btn-pause');
    const btnResume = document.getElementById('btn-resume');
    if (btnPause && btnResume && state) {
        if (state === 'PAUSED') {
            btnPause.classList.add('opacity-40', 'pointer-events-none');
            btnPause.disabled = true;
            btnResume.classList.remove('opacity-40', 'pointer-events-none');
            btnResume.disabled = false;
        } else {
            btnPause.classList.remove('opacity-40', 'pointer-events-none');
            btnPause.disabled = false;
            btnResume.classList.add('opacity-40', 'pointer-events-none');
            btnResume.disabled = true;
        }
    }
}

function updatePortfolioUI(portfolio) {
    if (!portfolio) return;
    // Unwrap nested portfolio object if present
    if (portfolio.portfolio && typeof portfolio.portfolio === 'object') {
        portfolio = portfolio.portfolio;
    }

    // Dynamic Regime Subtext
    const regime = portfolio.regime || portfolio.current_regime || AppState.current_regime;
    if (regime) {
        AppState.current_regime = regime;
        const regimeBadge = document.getElementById('regime-badge');
        if (regimeBadge) {
            regimeBadge.textContent = regime;
        }
        updateRegimeDesc(regime);
    }

    // 1. Total NAV
    const navEl = document.getElementById('portfolio-nav');
    if (navEl && typeof portfolio.total_nav === 'number') {
        navEl.textContent = formatCurrency(portfolio.total_nav);
    }

    // 2. Cash & Equity
    const cashEl = document.getElementById('portfolio-cash');
    if (cashEl && typeof portfolio.cash === 'number') {
        cashEl.textContent = formatCurrency(portfolio.cash);
    }

    const equityEl = document.getElementById('portfolio-equity');
    if (equityEl && typeof portfolio.equity === 'number') {
        equityEl.textContent = formatCurrency(portfolio.equity);
    }

    // 3. Asset Split Bar
    if (typeof portfolio.total_nav === 'number' && portfolio.total_nav > 0) {
        const cash = typeof portfolio.cash === 'number' ? portfolio.cash : 0;
        const equity = typeof portfolio.equity === 'number' ? portfolio.equity : 0;
        const cashPct = Math.min(100, Math.max(0, (cash / portfolio.total_nav) * 100));
        const eqPct = Math.min(100, Math.max(0, (equity / portfolio.total_nav) * 100));
        const barCash = document.getElementById('bar-cash');
        const barEq = document.getElementById('bar-equity');
        if (barCash) barCash.style.width = `${cashPct}%`;
        if (barEq) barEq.style.width = `${eqPct}%`;
        
        const cashWgt = document.getElementById('cash-weight');
        if (cashWgt) cashWgt.textContent = `${cashPct.toFixed(1)}% allocation`;
    }

    // 4. Unrealized P&L Badge
    const pnlBadge = document.getElementById('today-pnl-badge');
    if (pnlBadge && typeof portfolio.unrealized_pnl === 'number') {
        const pnl = portfolio.unrealized_pnl;
        const sign = pnl >= 0 ? '+' : '';
        pnlBadge.textContent = `${sign}${formatCurrency(pnl)}`;
        if (pnl >= 0) {
            pnlBadge.className = 'px-2 py-0.5 rounded-md text-xs font-bold font-mono bg-emerald-50 text-emerald-700 border border-emerald-200';
        } else {
            pnlBadge.className = 'px-2 py-0.5 rounded-md text-xs font-bold font-mono bg-rose-50 text-rose-700 border border-rose-200';
        }
    }

    // 5. Positions List (Mobile Card Stack & Desktop Table)
    const positions = portfolio.positions || [];
    const countEl = document.getElementById('positions-count');
    const badgeEl = document.getElementById('holdings-badge');
    if (countEl) countEl.textContent = `${positions.length} active position${positions.length === 1 ? '' : 's'}`;
    if (badgeEl) badgeEl.textContent = `${positions.length} Open`;

    renderPositions(positions, portfolio.total_nav || 50000.0);
}

function renderPositions(positions, totalNav) {
    const mobileContainer = document.getElementById('positions-mobile-list');
    const desktopTbody = document.getElementById('positions-body');

    if (!positions || positions.length === 0) {
        if (mobileContainer) {
            mobileContainer.innerHTML = `
                <div class="p-6 text-center text-slate-500 text-xs border border-dashed border-slate-200 rounded-xl bg-slate-50/50">
                    <div class="text-2xl mb-1 select-none">📦</div>
                    <p class="font-medium text-slate-600">No open positions</p>
                    <p class="text-[11px] text-slate-500 mt-0.5">$50,000.00 pristine cash balance in SHV / USD</p>
                </div>
            `;
        }
        if (desktopTbody) {
            desktopTbody.innerHTML = `
                <tr>
                    <td colspan="7" class="py-6 text-center text-slate-500">
                        No open positions ($50,000.00 cash pristine)
                    </td>
                </tr>
            `;
        }
        return;
    }

    // Render Mobile Cards
    if (mobileContainer) {
        mobileContainer.innerHTML = positions.map(pos => {
            const pnl = pos.unrealized_pnl || 0.0;
            const isProfit = pnl >= 0;
            const pnlClass = isProfit ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-rose-700 bg-rose-50 border-rose-200';
            const sign = isProfit ? '+' : '';
            const weightPct = ((pos.market_value / (totalNav || 1)) * 100).toFixed(1);

            return `
                <div class="p-3 bg-slate-50/70 border border-slate-200/80 rounded-xl space-y-2">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-2">
                            <span class="font-bold text-xs text-slate-900 bg-white border border-slate-200 px-2 py-0.5 rounded shadow-2xs">${pos.symbol}</span>
                            <span class="text-[11px] font-medium text-slate-500">${weightPct}% wt</span>
                        </div>
                        <span class="px-2 py-0.5 rounded text-[11px] font-bold border font-mono ${pnlClass}">
                            ${sign}${formatCurrency(pnl)} (${sign}${(pos.unrealized_pnl_pct * 100 || 0).toFixed(1)}%)
                        </span>
                    </div>
                    <div class="flex items-center justify-between text-xs text-slate-600 font-mono pt-1 border-t border-slate-200/50">
                        <span>${pos.shares || pos.qty} sh @ ${formatCurrency(pos.current_price || pos.avg_entry_price)}</span>
                        <span class="font-bold text-slate-800">${formatCurrency(pos.market_value)}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    // Render Desktop Table Rows
    if (desktopTbody) {
        desktopTbody.innerHTML = positions.map(pos => {
            const pnl = pos.unrealized_pnl || 0.0;
            const isProfit = pnl >= 0;
            const pnlClass = isProfit ? 'text-emerald-600' : 'text-rose-600';
            const sign = isProfit ? '+' : '';
            const weightPct = ((pos.market_value / (totalNav || 1)) * 100).toFixed(1);

            return `
                <tr class="hover:bg-slate-50/50 transition-colors">
                    <td class="py-2.5 font-bold text-slate-900">${pos.symbol}</td>
                    <td class="py-2.5 text-right font-mono">${pos.shares || pos.qty}</td>
                    <td class="py-2.5 text-right font-mono">${formatCurrency(pos.avg_entry_price)}</td>
                    <td class="py-2.5 text-right font-mono">${formatCurrency(pos.current_price || pos.avg_entry_price)}</td>
                    <td class="py-2.5 text-right font-mono font-semibold text-slate-900">${formatCurrency(pos.market_value)}</td>
                    <td class="py-2.5 text-right font-mono">${weightPct}%</td>
                    <td class="py-2.5 text-right font-mono font-semibold ${pnlClass}">
                        ${sign}${formatCurrency(pnl)}
                    </td>
                </tr>
            `;
        }).join('');
    }
}

// =============================================================================
// SSE Connection & Fallback
// =============================================================================
function initSSE() {
    if (window.EventSource === undefined) {
        console.warn("SSE not supported. Falling back to polling.");
        startPolling();
        return;
    }

    try {
        AppState.eventSource = new EventSource('/api/events');

        AppState.eventSource.onopen = () => {
            console.log("SSE Stream connected.");
            AppState.sseConnected = true;
            AppState.reconnectAttempts = 0;
            stopPolling();
            setSSEIndicator('connected');
        };

        const handleIncomingPayload = (rawData) => {
            try {
                const data = JSON.parse(rawData);
                updateStatusUI(data);
                updatePortfolioUI(data.portfolio || data);
            } catch (err) {
                console.error("Failed to parse SSE payload", err);
            }
        };

        AppState.eventSource.onmessage = (e) => {
            handleIncomingPayload(e.data);
        };

        AppState.eventSource.addEventListener('heartbeat', (e) => {
            handleIncomingPayload(e.data);
            const ping = document.getElementById('sse-ping');
            if (ping) {
                ping.classList.remove('opacity-0');
                setTimeout(() => ping.classList.add('opacity-0'), 500);
            }
        });

        AppState.eventSource.addEventListener('status', (e) => {
            handleIncomingPayload(e.data);
        });

        AppState.eventSource.addEventListener('portfolio', (e) => {
            handleIncomingPayload(e.data);
        });

        AppState.eventSource.addEventListener('rebalance', (e) => {
            handleIncomingPayload(e.data);
            fetchTrades();
        });

        AppState.eventSource.onerror = () => {
            console.warn("SSE Connection lost. Attempting recovery...");
            AppState.sseConnected = false;
            setSSEIndicator('reconnecting');
            AppState.eventSource.close();

            AppState.reconnectAttempts++;
            if (AppState.reconnectAttempts > 2) {
                startPolling();
            }

            const delay = Math.min(10000, Math.pow(2, AppState.reconnectAttempts) * 1000);
            setTimeout(initSSE, delay);
        };
    } catch (e) {
        console.error("SSE initialization exception", e);
        startPolling();
    }
}

function setSSEIndicator(status) {
    const ping = document.getElementById('sse-ping');
    const dot = document.getElementById('sse-dot');
    const label = document.getElementById('sse-label');

    if (status === 'connected') {
        if (dot) dot.className = 'relative inline-flex rounded-full h-2 w-2 bg-emerald-500';
        if (ping) ping.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75';
        if (label) {
            label.textContent = 'LIVE';
            label.className = 'text-[10px] sm:text-xs font-semibold text-emerald-700';
        }
    } else if (status === 'polling') {
        if (dot) dot.className = 'relative inline-flex rounded-full h-2 w-2 bg-amber-500';
        if (ping) ping.className = 'hidden';
        if (label) {
            label.textContent = 'POLL';
            label.className = 'text-[10px] sm:text-xs font-semibold text-amber-700';
        }
    } else {
        if (dot) dot.className = 'relative inline-flex rounded-full h-2 w-2 bg-rose-500';
        if (ping) ping.className = 'hidden';
        if (label) {
            label.textContent = 'OFFLINE';
            label.className = 'text-[10px] sm:text-xs font-semibold text-rose-700';
        }
    }
}

function startPolling() {
    if (AppState.pollingInterval) return;
    console.log("Starting polling fallback (every 3s)...");
    setSSEIndicator('polling');
    fetchSnapshot();
    AppState.pollingInterval = setInterval(fetchSnapshot, 3000);
}

function stopPolling() {
    if (AppState.pollingInterval) {
        clearInterval(AppState.pollingInterval);
        AppState.pollingInterval = null;
    }
}

async function fetchSnapshot() {
    try {
        const [pRes, sRes] = await Promise.all([
            fetch('/api/portfolio'),
            fetch('/api/status')
        ]);
        if (pRes.ok) {
            const pData = await pRes.json();
            updatePortfolioUI(pData.portfolio || pData);
        }
        if (sRes.ok) {
            const sData = await sRes.json();
            updateStatusUI(sData);
        }
    } catch (e) {
        console.error("Polling snapshot fetch failed", e);
        setSSEIndicator('disconnected');
    }
}

// =============================================================================
// Operator Action Handlers
// =============================================================================
function showFeedback(msg, isError = false) {
    const fb = document.getElementById('operator-feedback');
    if (!fb) return;
    if (AppState.feedbackTimer) {
        clearTimeout(AppState.feedbackTimer);
        AppState.feedbackTimer = null;
    }
    fb.textContent = msg;
    fb.className = isError 
        ? 'mt-3 p-2.5 rounded-lg text-xs font-semibold text-center bg-rose-50 text-rose-700 border border-rose-200 block'
        : 'mt-3 p-2.5 rounded-lg text-xs font-semibold text-center bg-emerald-50 text-emerald-700 border border-emerald-200 block';
    AppState.feedbackTimer = setTimeout(() => {
        fb.classList.add('hidden');
    }, 5000);
}


// Operator token: open the dashboard once as /#token=SECRET and it is kept in
// this browser. Viewing stays public; control buttons need the token.
function operatorHeaders() {
    try {
        const m = location.hash.match(/token=([^&]+)/);
        if (m) {
            localStorage.setItem('operatorToken', decodeURIComponent(m[1]));
            history.replaceState(null, '', location.pathname + location.search);
        }
        return { 'X-Operator-Token': localStorage.getItem('operatorToken') || '' };
    } catch (e) {
        return {};
    }
}

async function operatorPause() {
    if (AppState.inFlightAction) return;
    AppState.inFlightAction = true;
    const btn = document.getElementById('btn-pause');
    const orig = btn.innerHTML;
    btn.innerHTML = '<span>⏳ Pausing...</span>';
    btn.disabled = true;

    try {
        const res = await fetch('/api/operator/pause', { method: 'POST', headers: operatorHeaders() });
        const json = await res.json();
        if (res.ok) {
            updateStatusUI({ state: 'PAUSED' });
            showFeedback("Strategy daemon PAUSED. Rebalancing evaluations frozen.");
        } else {
            showFeedback(json.detail || "Pause failed", true);
        }
    } catch (e) {
        showFeedback("Network error during pause", true);
    } finally {
        btn.innerHTML = orig;
        AppState.inFlightAction = false;
        const currentBadge = document.getElementById('status-badge');
        const currentState = currentBadge ? currentBadge.textContent.trim() : null;
        btn.disabled = (currentState === 'PAUSED');
    }
}

async function operatorResume() {
    if (AppState.inFlightAction) return;
    AppState.inFlightAction = true;
    const btn = document.getElementById('btn-resume');
    const orig = btn.innerHTML;
    btn.innerHTML = '<span>⏳ Resuming...</span>';
    btn.disabled = true;

    try {
        const res = await fetch('/api/operator/resume', { method: 'POST', headers: operatorHeaders() });
        const json = await res.json();
        if (res.ok) {
            updateStatusUI({ state: 'RUNNING' });
            showFeedback("Strategy daemon RESUMED. Scheduled evaluations active.");
        } else {
            showFeedback(json.detail || "Resume failed", true);
        }
    } catch (e) {
        showFeedback("Network error during resume", true);
    } finally {
        btn.innerHTML = orig;
        AppState.inFlightAction = false;
        const currentBadge = document.getElementById('status-badge');
        const currentState = currentBadge ? currentBadge.textContent.trim() : null;
        btn.disabled = (currentState === 'RUNNING');
    }
}

function confirmRebalanceModal() {
    const modal = document.getElementById('rebalance-modal');
    if (modal) modal.classList.remove('hidden');
}

function closeRebalanceModal() {
    const modal = document.getElementById('rebalance-modal');
    if (modal) modal.classList.add('hidden');
}

async function executeManualRebalance() {
    closeRebalanceModal();
    if (AppState.inFlightAction) return;
    AppState.inFlightAction = true;

    const btn = document.getElementById('btn-rebalance');
    const orig = btn.innerHTML;
    btn.innerHTML = '<span>⏳ Rebalancing...</span>';
    btn.disabled = true;

    try {
        const res = await fetch('/api/operator/rebalance', { method: 'POST', headers: operatorHeaders() });
        const json = await res.json();
        if (res.ok) {
            if (!json.success || (json.status && json.status.startsWith('REJECTED'))) {
                const reason = json.status_message || json.rationale || json.status || "Manual rebalance rejected";
                showFeedback(`Rebalance rejected: ${reason}`, true);
                return;
            }
            const count = json.orders_count || 0;
            if (count > 0) {
                showFeedback(`Rebalance executed: ${count} orders filled successfully.`);
            } else {
                showFeedback("Rebalance evaluated: All holdings within ±2.5% drift band. 0 orders needed.");
            }
            if (json.portfolio_state_after) {
                updatePortfolioUI(json.portfolio_state_after);
            } else {
                fetchSnapshot();
            }
            fetchTrades();
        } else {
            showFeedback(json.detail || "Manual rebalance rejected", true);
        }
    } catch (e) {
        showFeedback("Network error during rebalance", true);
    } finally {
        btn.innerHTML = orig;
        btn.disabled = false;
        AppState.inFlightAction = false;
    }
}

// =============================================================================
// Trade Activity Rendering
// =============================================================================
function formatTimestamp(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return isoStr;
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch (e) {
        return isoStr;
    }
}

function renderActivity(trades) {
    const container = document.getElementById('activity-list');
    if (!container) return;

    if (!trades || trades.length === 0) {
        container.innerHTML = `
            <p id="activity-empty" class="text-slate-500 text-center py-3">
                No trades executed yet. Initialized at $50,000.00 starting balance.
            </p>
        `;
        return;
    }

    // Render chronologically with newest trades at the top
    const sorted = [...trades].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    container.innerHTML = sorted.map(t => {
        const isBuy = (t.side || '').toUpperCase() === 'BUY';
        const badgeClass = isBuy 
            ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
            : 'bg-rose-50 text-rose-700 border-rose-200';
        const timeStr = formatTimestamp(t.timestamp);
        const shares = t.shares !== undefined ? t.shares : t.qty;
        const price = formatCurrency(t.price);

        return `
            <div class="p-2.5 bg-slate-50/70 border border-slate-200/80 rounded-xl flex items-center justify-between">
                <div class="flex items-center space-x-2">
                    <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass}">${t.side}</span>
                    <span class="font-bold text-xs text-slate-900">${t.symbol}</span>
                    <span class="text-[11px] text-slate-500">${shares} sh @ ${price}</span>
                </div>
                <span class="text-[11px] font-mono text-slate-500">${timeStr}</span>
            </div>
        `;
    }).join('');
}

async function fetchTrades() {
    try {
        const res = await fetch('/api/trades');
        if (res.ok) {
            const data = await res.json();
            renderActivity(data.trades || []);
        }
    } catch (e) {
        console.error("Failed to fetch trades activity", e);
    }
}

// =============================================================================
// Modal Dismissal & Keyboard Navigation
// =============================================================================
function initModalHandlers() {
    const modal = document.getElementById('rebalance-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            const card = modal.querySelector('.bg-white');
            if (card && !card.contains(e.target)) {
                closeRebalanceModal();
            } else if (e.target === modal) {
                closeRebalanceModal();
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' || e.key === 'Esc') {
            const m = document.getElementById('rebalance-modal');
            if (m && !m.classList.contains('hidden')) {
                closeRebalanceModal();
            }
        }
    });
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
    const banner = document.getElementById('alert-banner');
    if (banner && !banner.classList.contains('hidden')) {
        AppState.alert_banner_active = true;
        AppState.feed_source = 'synthetic_fallback';
    }
    initSSE();
    fetchSnapshot();
    fetchTrades();
    initModalHandlers();
});
