"""Banker's Algorithm Resource Management System (Streamlit)."""
import numpy as np
import pandas as pd
import streamlit as st

from banker import calculate_need, process_request, safety_check, validate_system

st.set_page_config(page_title="Banker's Algorithm Resource Management System",
                   page_icon="🏦", layout="wide")

# Default data (classic textbook example) used to pre-fill the input tables.
EX_ALLOC = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
EX_MAX = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
EX_AVAIL = [3, 3, 2]

INDIGO, AMBER, VIOLET, GREEN = (79, 70, 229), (245, 158, 11), (124, 58, 237), (16, 185, 129)

# ---------- styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
@keyframes fadeUp {from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;}}
@keyframes pop {0% {opacity:0; transform:scale(.6);} 70% {transform:scale(1.08);} 100% {opacity:1; transform:scale(1);}}
@keyframes pulse {0%,100% {box-shadow:0 0 0 0 rgba(16,185,129,.45);} 50% {box-shadow:0 0 0 12px rgba(16,185,129,0);}}
@keyframes pulseRed {0%,100% {box-shadow:0 0 0 0 rgba(239,68,68,.45);} 50% {box-shadow:0 0 0 12px rgba(239,68,68,0);}}
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%); border-radius:22px;
  padding:30px 34px; color:#fff; animation: fadeUp .6s ease both; box-shadow:0 12px 30px rgba(79,70,229,.28);}
.hero h1 {color:#fff; margin:0; font-size:2.1rem; line-height:1.2; padding:0;}
.hero p {margin:6px 0 0 0; opacity:.9; font-size:1.02rem;}
.flow {display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:18px;}
.flow span.s {background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28); border-radius:999px;
  padding:5px 13px; font-size:.82rem; backdrop-filter: blur(4px);}
.flow span.a {opacity:.7;}
div[data-baseweb="tab-list"] {gap:8px; background:#e9ebf6; padding:6px; border-radius:16px;}
button[data-baseweb="tab"] {border-radius:12px !important; padding:8px 16px !important; font-weight:600; height:auto;}
button[data-baseweb="tab"][aria-selected="true"] {background:#fff !important; color:#4f46e5 !important;
  box-shadow:0 2px 8px rgba(79,70,229,.18);}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display:none;}
.card {background:#fff; border:1px solid #e5e7f2; border-radius:18px; padding:16px 18px;
  box-shadow:0 4px 14px rgba(31,41,55,.05); animation: fadeUp .5s ease both;}
.card h4 {margin:0 0 4px 0; font-size:1rem;}
.sub {color:#6b7280; font-size:.86rem; margin-bottom:8px;}
.stat {background:#fff; border-radius:18px; padding:16px 18px; border:1px solid #e5e7f2;
  box-shadow:0 4px 14px rgba(31,41,55,.05); animation: fadeUp .5s ease both; border-top:4px solid var(--c);}
.stat .l {color:#6b7280; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em;}
.stat .v {font-size:1.9rem; font-weight:800; color:#111827; line-height:1.2;}
.stat .d {color:#6b7280; font-size:.8rem;}
.badge {display:flex; align-items:center; gap:14px; border-radius:20px; padding:18px 22px; color:#fff; font-weight:700;
  animation: fadeUp .5s ease both;}
.badge.safe {background:linear-gradient(120deg,#059669,#10b981); animation: pulse 2.4s infinite;}
.badge.unsafe {background:linear-gradient(120deg,#dc2626,#f97316); animation: pulseRed 2.4s infinite;}
.badge .ic {font-size:2rem;}
.badge .t {font-size:1.25rem;}
.badge .s {font-weight:500; opacity:.95; font-size:.95rem;}
.chip {display:inline-block; background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe; padding:6px 15px;
  border-radius:999px; font-weight:800; animation: pop .45s ease both;}
.chip.ok {background:#dcfce7; color:#047857; border-color:#86efac;}
.arrow {color:#9ca3af; margin:0 6px; font-weight:700;}
.vec {display:inline-flex; gap:4px; vertical-align:middle;}
.vec b {background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px; min-width:26px; text-align:center;
  padding:1px 6px; font-weight:700; font-size:.9rem; color:#111827;}
.vec.g b {background:#dcfce7; border-color:#86efac; color:#047857;}
.vec.a b {background:#fef3c7; border-color:#fcd34d; color:#92400e;}
.vec.i b {background:#e0e7ff; border-color:#a5b4fc; color:#3730a3;}
.vec.r b {background:#fee2e2; border-color:#fca5a5; color:#991b1b;}
.step {display:flex; gap:14px; align-items:flex-start; background:#fff; border:1px solid #e5e7f2; border-radius:16px;
  padding:14px 16px; margin-bottom:10px; animation: fadeUp .5s ease both; border-left:5px solid #10b981;}
.step.bad {border-left-color:#ef4444;}
.step .n {background:#10b981; color:#fff; border-radius:50%; min-width:34px; height:34px; display:flex;
  align-items:center; justify-content:center; font-weight:800;}
.step.bad .n {background:#ef4444;}
.step .body {line-height:1.9; font-size:.93rem;}
.chk {display:flex; align-items:center; gap:10px; padding:9px 12px; border-radius:12px; margin-bottom:8px;
  font-size:.92rem; font-weight:600;}
.chk.ok {background:#ecfdf5; color:#047857;} .chk.no {background:#fef2f2; color:#b91c1c;}
.chk.wait {background:#f3f4f6; color:#6b7280;}
.bar {background:#e5e7eb; height:12px; border-radius:99px; overflow:hidden;}
.bar > div {height:100%; border-radius:99px; background:linear-gradient(90deg,#6366f1,#a855f7);}
.empty {text-align:center; padding:38px 20px; background:#fff; border:2px dashed #c7d2fe; border-radius:20px;
  color:#4338ca; font-weight:600; animation: fadeUp .5s ease both;}
.empty .big {font-size:2.4rem;}
div.stButton > button {border-radius:14px; font-weight:700; padding:.55rem 1.4rem; transition:all .2s ease;}
div.stButton > button:hover {transform:translateY(-2px); box-shadow:0 8px 18px rgba(79,70,229,.25);}
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------
def pname(i): return f"P{i}"
def rname(j): return chr(ord("A") + j) if j < 26 else f"R{j}"


def frame(arr, procs=True):
    arr = np.asarray(arr)
    if procs:
        return pd.DataFrame(arr, index=[pname(i) for i in range(arr.shape[0])],
                            columns=[rname(j) for j in range(arr.shape[1])])
    return pd.DataFrame([arr], index=["Available"],
                        columns=[rname(j) for j in range(arr.shape[0])])


def heat(df, rgb):
    """Colour-scaled table: bigger numbers get stronger colour."""
    vmax = max(int(df.to_numpy().max()), 1) if df.size else 1
    r, g, b = rgb

    def f(d):
        return pd.DataFrame(
            [[f"background-color: rgba({r},{g},{b},{0.07 + 0.5 * int(v) / vmax:.2f}); color:#111827; "
              f"font-weight:700; text-align:center" for v in row] for row in d.to_numpy()],
            index=d.index, columns=d.columns)
    return df.style.apply(f, axis=None)


def show(arr, rgb, procs=True):
    st.dataframe(heat(frame(arr, procs), rgb), width="stretch")


def vec(v, cls=""):
    return f'<span class="vec {cls}">' + "".join(f"<b>{int(x)}</b>" for x in v) + "</span>"


def fmt_vec(v): return "(" + ", ".join(str(int(x)) for x in v) + ")"


def seq_text(seq): return " → ".join(pname(i) for i in seq)


def chips(seq, ok=True):
    out = ""
    for k, i in enumerate(seq):
        if k:
            out += '<span class="arrow">→</span>'
        out += f'<span class="chip {"ok" if ok else ""}" style="animation-delay:{k * 0.12:.2f}s">{pname(i)}</span>'
    return out


def stat(label, value, sub, color):
    return (f'<div class="stat" style="--c:{color}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="d">{sub}</div></div>')


def card(title, sub=""):
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def need_system():
    if "sys" not in st.session_state:
        st.markdown('<div class="empty"><div class="big">🧩</div>No system loaded yet.<br>'
                    'Go to the <b>Process Input</b> tab, fill in the tables and press <b>Save System</b>.</div>',
                    unsafe_allow_html=True)
        return None
    return st.session_state["sys"]


def status_badge(r):
    if r.is_safe:
        st.markdown(f'<div class="badge safe"><div class="ic">🛡️</div><div><div class="t">System is SAFE</div>'
                    f'<div class="s">Safe sequence: {seq_text(r.sequence)}</div></div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge unsafe"><div class="ic">🚨</div><div><div class="t">System is UNSAFE</div>'
                    '<div class="s">No safe sequence exists — deadlock is possible.</div></div></div>',
                    unsafe_allow_html=True)


# ---------- hero ----------
flow = ["Enter process & resource details", "Calculate Need Matrix", "Receive resource request",
        "Check available resources", "Perform safety algorithm", "Allocate resources if safe",
        "Display safe sequence & status"]
flow_html = '<span class="a">→</span>'.join(f'<span class="s">{k + 1}. {t}</span>' for k, t in enumerate(flow))
st.markdown(f'<div class="hero"><h1>🏦 Banker\'s Algorithm<br>Resource Management System</h1>'
            f'<p>Allocate resources only when the system stays safe — deadlock avoidance, simulated live.</p>'
            f'<div class="flow">{flow_html}</div></div>', unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["① Process Input", "② Need Calculation", "③ Request & Allocation",
                "④ Safety Check", "⑤ Status Display"])

# ---------- 1. Process input ----------
with tabs[0]:
    st.markdown("### 📥 Process Input")
    st.caption("Enter processes, available resources, allocated resources, and maximum resource requirements.")
    c1, c2 = st.columns(2)
    n = int(c1.number_input("🧮 Number of processes", 1, 10, 5, key="n_proc"))
    m = int(c2.number_input("🧱 Number of resource types", 1, 6, 3, key="n_res"))

    use_example = (n, m) == (5, 3)
    alloc_df = frame(EX_ALLOC if use_example else np.zeros((n, m), dtype=int))
    max_df = frame(EX_MAX if use_example else np.zeros((n, m), dtype=int))
    avail_df = frame(EX_AVAIL if use_example else np.zeros(m, dtype=int), procs=False)

    cfg = {c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d")
           for c in alloc_df.columns}
    key = f"{n}x{m}"
    ca, cb = st.columns(2)
    with ca:
        card("🟦 Allocation Matrix", "What each process holds right now")
        alloc_ed = st.data_editor(alloc_df, column_config=cfg, key=f"alloc_{key}", width="stretch")
    with cb:
        card("🟪 Maximum Matrix", "The most each process will ever need")
        max_ed = st.data_editor(max_df, column_config=cfg, key=f"max_{key}", width="stretch")
    card("🟩 Available Resources", "Free instances of each resource type")
    avail_ed = st.data_editor(avail_df, column_config=cfg, key=f"avail_{key}", width="stretch")

    if st.button("💾 Save System", type="primary"):
        try:
            alloc = alloc_ed.to_numpy(dtype=float)
            mx = max_ed.to_numpy(dtype=float)
            av = avail_ed.to_numpy(dtype=float)[0]
            if np.isnan(alloc).any() or np.isnan(mx).any() or np.isnan(av).any():
                raise ValueError("All cells must be filled in.")
            if (alloc % 1).any() or (mx % 1).any() or (av % 1).any():
                raise ValueError("Please enter whole numbers only.")
            alloc, mx, av = alloc.astype(int), mx.astype(int), av.astype(int)
            err = validate_system(av, alloc, mx)
            if err:
                raise ValueError(err)
            st.session_state["sys"] = {"alloc": alloc, "max": mx, "avail": av,
                                       "need": calculate_need(mx, alloc)}
            st.session_state["log"] = []
            st.session_state.pop("last", None)
            st.toast("System saved!", icon="✅")
            st.success("System saved. Continue to the next tabs →")
        except ValueError as e:
            st.error(str(e))

# ---------- 2. Need calculation ----------
with tabs[1]:
    st.markdown("### 🧮 Need Calculation")
    s = need_system()
    if s:
        st.markdown('<div class="card"><h4>Need = Maximum − Allocation</h4>'
                    '<div class="sub">The remaining resources each process still requires.</div></div>',
                    unsafe_allow_html=True)
        st.write("")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**🟪 Maximum**"); show(s["max"], VIOLET)
        with b:
            st.markdown("**🟦 Allocation**"); show(s["alloc"], INDIGO)
        with c:
            st.markdown("**🟧 Need**"); show(s["need"], AMBER)

# ---------- 3. Resource request & allocation ----------
with tabs[2]:
    st.markdown("### 📨 Resource Request & Allocation")
    s = need_system()
    if s:
        st.caption("Resources are allocated only when the system remains in a safe state.")
        n_p, n_r = s["alloc"].shape
        left, right = st.columns([1.1, 1])
        with left:
            pid = st.selectbox("👤 Requesting process", range(n_p), format_func=pname, key="req_pid")
            st.markdown(f'Need of **{pname(pid)}** {vec(s["need"][pid], "a")} &nbsp;·&nbsp; '
                        f'Available {vec(s["avail"], "g")}', unsafe_allow_html=True)
            cols = st.columns(n_r)
            req = np.array([int(cols[j].number_input(f"Resource {rname(j)}", min_value=0, step=1,
                                                     key=f"req_{pid}_{j}")) for j in range(n_r)])
            clicked = st.button("🚀 Submit Request", type="primary")
        with right:
            preview = st.container()

        if clicked:
            res, av, al, nd = process_request(pid, req, s["avail"], s["alloc"], s["need"])
            entry = {"process": pname(pid), "request": fmt_vec(req),
                     "result": "Granted" if res.granted else "Denied", "reason": res.message}
            if res.granted:
                s.update(avail=av, alloc=al, need=nd)
                entry["sequence"] = seq_text(res.safety.sequence)
                st.toast(f"{pname(pid)} request granted", icon="✅")
                st.markdown(f'<div class="badge safe"><div class="ic">✅</div><div><div class="t">Request GRANTED</div>'
                            f'<div class="s">{res.message}</div></div></div>', unsafe_allow_html=True)
                st.markdown("**Safe sequence after allocation**")
                st.markdown(chips(res.safety.sequence), unsafe_allow_html=True)
            else:
                st.toast(f"{pname(pid)} request denied", icon="⛔")
                st.markdown(f'<div class="badge unsafe"><div class="ic">⛔</div><div><div class="t">Request DENIED</div>'
                            f'<div class="s">{res.message}</div></div></div>', unsafe_allow_html=True)
            st.session_state["log"].append(entry)
            st.session_state["last"] = entry
        else:
            # Live pre-check of the three conditions while the user types.
            need_row, avail_row = s["need"][pid], s["avail"]
            empty = not np.any(req > 0)
            c1 = bool(np.all(req <= need_row))
            c2 = bool(np.all(req <= avail_row))
            c3 = None
            if not empty and c1 and c2:
                ta, tl, tn = avail_row - req, s["alloc"].copy(), s["need"].copy()
                tl[pid] += req; tn[pid] -= req
                c3 = safety_check(ta, tl, tn).is_safe

            def row(state, text):
                cls, icon = {True: ("ok", "✅"), False: ("no", "❌"), None: ("wait", "⏳")}[state]
                return f'<div class="chk {cls}"><span>{icon}</span><span>{text}</span></div>'

            with preview:
                st.markdown(
                    '<div class="card"><h4>🔍 Live pre-check</h4><div class="sub">Updates as you type — '
                    'nothing is allocated until you submit.</div>'
                    + row(None if empty else c1, "Request ≤ process Need")
                    + row(None if empty else c2, "Request ≤ Available")
                    + row(c3, "System stays SAFE after allocation") + '</div>',
                    unsafe_allow_html=True)

# ---------- 4. Safety check ----------
with tabs[3]:
    st.markdown("### 🛡️ Safety Check")
    s = need_system()
    if s:
        st.caption("Runs the safety algorithm on the current system state: "
                   "find a process whose Need ≤ Work, let it finish, and add its Allocation back to Work.")
        if st.button("▶️ Run Safety Algorithm", type="primary"):
            r = safety_check(s["avail"], s["alloc"], s["need"])
            status_badge(r)
            st.write("")
            st.markdown(f"**Starting Work = Available** {vec(s['avail'], 'g')}", unsafe_allow_html=True)
            for k, x in enumerate(r.steps):
                st.markdown(
                    f'<div class="step" style="animation-delay:{k * 0.1:.2f}s"><div class="n">{k + 1}</div>'
                    f'<div class="body"><b>{pname(x["process"])}</b> can finish &nbsp;·&nbsp; '
                    f'Need {vec(x["need"], "a")} ≤ Work {vec(x["work_before"], "g")}<br>'
                    f'It releases its resources → new Work {vec(x["work_after"], "i")}</div></div>',
                    unsafe_allow_html=True)
            if not r.is_safe:
                work = r.steps[-1]["work_after"] if r.steps else s["avail"]
                for i in range(s["alloc"].shape[0]):
                    if i not in r.sequence:
                        st.markdown(
                            f'<div class="step bad"><div class="n">✕</div><div class="body">'
                            f'<b>{pname(i)}</b> is stuck &nbsp;·&nbsp; Need {vec(s["need"][i], "r")} '
                            f'cannot be met by Work {vec(work, "g")}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown("**Safe sequence**")
                st.markdown(chips(r.sequence), unsafe_allow_html=True)

# ---------- 5. Status display ----------
with tabs[4]:
    st.markdown("### 📊 Status Display")
    s = need_system()
    if s:
        r = safety_check(s["avail"], s["alloc"], s["need"])
        status_badge(r)
        if r.is_safe:
            st.markdown(chips(r.sequence), unsafe_allow_html=True)
        st.write("")

        total_alloc = int(s["alloc"].sum())
        total_av = int(s["avail"].sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(stat("Processes", s["alloc"].shape[0], "running in the system", "#6366f1"), unsafe_allow_html=True)
        k2.markdown(stat("Resource types", s["alloc"].shape[1], "A, B, C …", "#a855f7"), unsafe_allow_html=True)
        k3.markdown(stat("Allocated", total_alloc, "instances in use", "#3b82f6"), unsafe_allow_html=True)
        k4.markdown(stat("Available", total_av, "instances free", "#10b981"), unsafe_allow_html=True)
        st.write("")

        st.markdown("**Resource usage**")
        bars = ""
        for j in range(s["alloc"].shape[1]):
            used = int(s["alloc"][:, j].sum()); free = int(s["avail"][j]); tot = used + free
            pct = 100 * used / tot if tot else 0
            bars += (f'<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;'
                     f'font-size:.88rem"><b>Resource {rname(j)}</b><span>{used} used / {tot} total</span></div>'
                     f'<div class="bar"><div style="width:{pct:.0f}%"></div></div></div>')
        st.markdown(f'<div class="card">{bars}</div>', unsafe_allow_html=True)
        st.write("")

        a, b = st.columns(2)
        with a:
            st.markdown("**🟦 Allocated Resources**"); show(s["alloc"], INDIGO)
        with b:
            st.markdown("**🟩 Available Resources**"); show(s["avail"], GREEN, procs=False)
            st.markdown("**🟧 Need Matrix**"); show(s["need"], AMBER)

        if st.session_state.get("log"):
            st.markdown("**🕘 Request History**")
            hist = pd.DataFrame(st.session_state["log"])
            hist = hist.style.map(
                lambda v: "background-color:#dcfce7;color:#047857;font-weight:700" if v == "Granted"
                else ("background-color:#fee2e2;color:#b91c1c;font-weight:700" if v == "Denied" else ""),
                subset=["result"])
            st.dataframe(hist, hide_index=True, width="stretch")
