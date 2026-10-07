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


def fmt_vec(v): return "(" + ", ".join(str(int(x)) for x in v) + ")"


def seq_text(seq): return " → ".join(pname(i) for i in seq)


def need_system():
    if "sys" not in st.session_state:
        st.info("Please enter and save the process details in the **Process Input** tab first.")
        return None
    return st.session_state["sys"]


# ---------- header ----------
st.title("🏦 Banker's Algorithm Resource Management System")
st.caption("A simulation of deadlock avoidance using the Banker's Algorithm")

tabs = st.tabs(["📄 Project Info", "1️⃣ Process Input", "2️⃣ Need Calculation",
                "3️⃣ Resource Request & Allocation", "4️⃣ Safety Check", "5️⃣ Status Display"])

# ---------- Project info (content from the report) ----------
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Abstract")
        st.write("A simulation-based system that manages resource allocation among multiple "
                 "processes using the Banker's Algorithm. The project demonstrates how Operating "
                 "System concepts can be used to prevent deadlocks by checking whether resource "
                 "allocation keeps the system in a safe state.")
        st.subheader("Objectives")
        st.markdown("- Prevent deadlock during resource allocation.\n"
                    "- Check whether the system is in a safe state.\n"
                    "- Allocate resources efficiently among processes.\n"
                    "- Demonstrate the Banker's Algorithm in a real-world scenario.")
        st.subheader("Major Modules")
        st.markdown(
            "- **Process Input:** Enter processes, available resources, allocated resources, and maximum resource requirements.\n"
            "- **Need Calculation:** Calculate the remaining resources required by each process.\n"
            "- **Resource Request:** Receive and verify resource requests from processes.\n"
            "- **Safety Check:** Check whether the requested allocation keeps the system in a safe state.\n"
            "- **Resource Allocation:** Allocate resources only when the system remains safe.\n"
            "- **Status Display:** Display allocated resources, available resources, and safe/unsafe status.")
        st.subheader("Methodology")
        st.info("Enter process and resource details → Calculate Need Matrix → Receive resource request "
                "→ Check available resources → Perform safety algorithm → Allocate resources if safe "
                "→ Display safe sequence and status.")
    with c2:
        st.subheader("Team Members")
        st.markdown("**E Samitha** — RA2511026020395  \n**R Swetha** — RA2511026020401")
        st.markdown("**Year:** II  \n**Semester:** III")
        st.subheader("Technologies")
        st.markdown("- **Python** — Main programming language\n"
                    "- **Streamlit** — Web-based GUI\n"
                    "- **NumPy** — Matrix calculations and resource management\n"
                    "- **VS Code** — Development environment")
        st.subheader("Expected Outcome")
        st.write("A working simulation that demonstrates deadlock avoidance using the Banker's "
                 "Algorithm and efficiently manages resource allocation among multiple processes.")

# ---------- 1. Process input ----------
with tabs[1]:
    st.subheader("Process Input")
    st.write("Enter the processes, available resources, allocated resources, and maximum resource requirements.")
    c1, c2 = st.columns(2)
    n = int(c1.number_input("Number of processes", 1, 10, 5, key="n_proc"))
    m = int(c2.number_input("Number of resource types", 1, 6, 3, key="n_res"))

    use_example = (n, m) == (5, 3)
    alloc_df = frame(EX_ALLOC if use_example else np.zeros((n, m), dtype=int))
    max_df = frame(EX_MAX if use_example else np.zeros((n, m), dtype=int))
    avail_df = frame(EX_AVAIL if use_example else np.zeros(m, dtype=int), procs=False)

    cfg = {c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d")
           for c in alloc_df.columns}
    key = f"{n}x{m}"
    st.markdown("**Allocation Matrix**")
    alloc_ed = st.data_editor(alloc_df, column_config=cfg, key=f"alloc_{key}", width="stretch")
    st.markdown("**Maximum Matrix**")
    max_ed = st.data_editor(max_df, column_config=cfg, key=f"max_{key}", width="stretch")
    st.markdown("**Available Resources**")
    avail_ed = st.data_editor(avail_df, column_config=cfg, key=f"avail_{key}", width="stretch")

    if st.button("Save System", type="primary"):
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
            st.success("System saved. Continue to the next tabs.")
        except ValueError as e:
            st.error(str(e))

# ---------- 2. Need calculation ----------
with tabs[2]:
    st.subheader("Need Calculation")
    s = need_system()
    if s:
        st.write("Need = Maximum − Allocation (remaining resources required by each process).")
        a, b, c = st.columns(3)
        a.markdown("**Maximum**"); a.dataframe(frame(s["max"]), width="stretch")
        b.markdown("**Allocation**"); b.dataframe(frame(s["alloc"]), width="stretch")
        c.markdown("**Need**"); c.dataframe(frame(s["need"]), width="stretch")

# ---------- 3. Resource request & allocation ----------
with tabs[3]:
    st.subheader("Resource Request & Allocation")
    s = need_system()
    if s:
        st.write("Resources are allocated only when the system remains in a safe state.")
        n_p, n_r = s["alloc"].shape
        pid = st.selectbox("Requesting process", range(n_p), format_func=pname, key="req_pid")
        cols = st.columns(n_r)
        req = np.array([int(cols[j].number_input(f"Resource {rname(j)}", min_value=0, step=1,
                                                 key=f"req_{pid}_{j}")) for j in range(n_r)])
        st.caption(f"Current Need of {pname(pid)}: {fmt_vec(s['need'][pid])}  |  "
                   f"Available: {fmt_vec(s['avail'])}")

        if st.button("Submit Request", type="primary"):
            res, av, al, nd = process_request(pid, req, s["avail"], s["alloc"], s["need"])
            entry = {"process": pname(pid), "request": fmt_vec(req),
                     "result": "Granted" if res.granted else "Denied", "reason": res.message}
            if res.granted:
                s.update(avail=av, alloc=al, need=nd)
                st.success(f"{res.message}  Safe sequence: {seq_text(res.safety.sequence)}")
                entry["sequence"] = seq_text(res.safety.sequence)
            else:
                st.error(res.message)
            st.session_state["log"].append(entry)
            st.session_state["last"] = entry

# ---------- 4. Safety check ----------
with tabs[4]:
    st.subheader("Safety Check")
    s = need_system()
    if s:
        st.write("Runs the safety algorithm on the current system state.")
        if st.button("Run Safety Algorithm", type="primary"):
            r = safety_check(s["avail"], s["alloc"], s["need"])
            if r.is_safe:
                st.success(f"✅ System is in a SAFE state.  Safe sequence: {seq_text(r.sequence)}")
            else:
                st.error("⚠️ System is in an UNSAFE state. No safe sequence exists.")
            if r.steps:
                st.markdown("**Execution steps**")
                st.dataframe(pd.DataFrame([{
                    "Step": k + 1, "Process": pname(x["process"]),
                    "Work (before)": fmt_vec(x["work_before"]),
                    "Need": fmt_vec(x["need"]),
                    "Work (after) = Work + Allocation": fmt_vec(x["work_after"]),
                } for k, x in enumerate(r.steps)]), hide_index=True, width="stretch")

# ---------- 5. Status display ----------
with tabs[5]:
    st.subheader("Status Display")
    s = need_system()
    if s:
        r = safety_check(s["avail"], s["alloc"], s["need"])
        if r.is_safe:
            st.success(f"✅ SAFE state — Safe sequence: {seq_text(r.sequence)}")
        else:
            st.error("⚠️ UNSAFE state — no safe sequence exists.")
        a, b = st.columns(2)
        a.markdown("**Allocated Resources**"); a.dataframe(frame(s["alloc"]), width="stretch")
        b.markdown("**Available Resources**"); b.dataframe(frame(s["avail"], procs=False), width="stretch")
        st.markdown("**Need Matrix**"); st.dataframe(frame(s["need"]), width="stretch")
        if st.session_state.get("log"):
            st.markdown("**Request History**")
            st.dataframe(pd.DataFrame(st.session_state["log"]), hide_index=True, width="stretch")
