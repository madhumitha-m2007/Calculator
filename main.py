
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Math Mega Calculator", layout="wide")
st.title("One Webpage")

tool = st.sidebar.selectbox(
    "Select Calculator",
    [
        "Cayley–Hamilton Theorem",
        "Maxima–Minima Calculator",
        "Lagrange Multiplier Calculator",
        "Double Integral (Polar Coordinates)",
        "Triple Integral with 3D Graph"
    ]
)

# --- Cayley Hamilton ---
if tool == "Cayley–Hamilton Theorem":
    st.header("📐 Cayley–Hamilton Calculator")
    matrix_input = st.text_area("Enter a square matrix:", "2 1\n1 2")
    if st.button("Compute Cayley–Hamilton"):
        try:
            rows = [r.strip() for r in matrix_input.split("\n") if r.strip()]
            mat = [[sp.sympify(e) for e in r.replace(",", " ").split()] for r in rows]
            A = sp.Matrix(mat)
            if A.rows != A.cols:
                st.error("Matrix must be square.")
            else:
                st.write(A)
                x = sp.symbols("x")
                charpoly = A.charpoly(x).as_expr()
                st.latex(sp.latex(charpoly))
                coeffs = A.charpoly().all_coeffs()
                n = A.rows
                pA = sp.zeros(n)
                for i,c in enumerate(coeffs):
                    power = len(coeffs)-i-1
                    pA += c*(A**power if power>0 else sp.eye(n))
                st.write(pA)
                if pA==sp.zeros(n): st.success("C-H Verified!")
        except Exception as e:
            st.error(e)

# --- Maxima Minima ---
if tool == "Maxima–Minima Calculator":
    st.header("📈 Maxima–Minima Calculator")
    func_input = st.text_input("Enter f(x):","x**3-3*x+2")
    if st.button("Compute"):
        try:
            x=sp.symbols('x')
            f=sp.sympify(func_input)
            f1=sp.diff(f,x)
            f2=sp.diff(f1,x)
            cps=sp.solve(f1,x)
            st.write("Critical Points:", cps)
            for cp in cps:
                sv=f2.subs(x,cp)
                if sv>0: st.success(f"Minimum at {cp}")
                elif sv<0: st.success(f"Maximum at {cp}")
                else: st.warning("Inconclusive.")
        except Exception as e:
            st.error(e)

# --- Lagrange ---
if tool == "Lagrange Multiplier Calculator":
    st.header("🧮 Lagrange Multiplier Calculator")

    allowed={"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"log":sp.log,"exp":sp.exp,"sqrt":sp.sqrt,"pi":sp.pi}
    var_count = st.radio("Variables:",[2,3])

    # ---- 2 variables ----
    if var_count==2:
        x,y,lam = sp.symbols("x y lam")
        f_str=st.text_input("f(x,y):","x**2 + y**2")
        g_str=st.text_input("g(x,y)=0:","x + y - 1")
        try:
            f=sp.sympify(f_str,allowed)
            g=sp.sympify(g_str,allowed)
            L=f+lam*g
            eq=[sp.diff(L,x),sp.diff(L,y),sp.diff(L,lam)]
            sol=sp.solve(eq,[x,y,lam],dict=True)
            formatted=[{str(k):str(v) for k,v in s.items()} for s in sol]
            st.json(formatted)
        except Exception as e:
            st.error(e)

    # ---- 3 variables ----
    else:
        x,y,z,lam=sp.symbols("x y z lam")
        f_str=st.text_input("f(x,y,z):","x**2 + y**2 + z**2")
        g_str=st.text_input("g(x,y,z)=0:","x+y+z-1")
        try:
            f=sp.sympify(f_str,allowed)
            g=sp.sympify(g_str,allowed)
            L=f+lam*g
            eq=[sp.diff(L,v) for v in (x,y,z,lam)]
            sol=sp.solve(eq,[x,y,z,lam],dict=True)
            formatted=[{str(k):str(v) for k,v in s.items()} for s in sol]
            st.json(formatted)
        except Exception as e:
            st.error(e)

# --- Double Integral ---
if tool == "Double Integral (Polar Coordinates)":
    st.header("🔵 Double Integral – Polar")
    safe={"sin":np.sin,"cos":np.cos,"tan":np.tan,"exp":np.exp,"sqrt":np.sqrt,"pi":np.pi}
    fstr=st.text_input("f(r,θ):","r*sin(theta)")
    rmin=st.text_input("r min:","0")
    rmax=st.text_input("r max:","1+cos(theta)")
    t1s=st.text_input("θ min:","0")
    t2s=st.text_input("θ max:","2*pi")
    n=st.slider("Resolution",200,2000,800)
    if st.button("Compute Integral"):
        try:
            f=lambda r,t: eval(fstr,{**safe,"r":r,"theta":t})
            rl=lambda t: eval(rmin,{**safe,"theta":t})
            ru=lambda t: eval(rmax,{**safe,"theta":t})
            t1=eval(t1s,safe)
            t2=eval(t2s,safe)
            thetas=np.linspace(t1,t2,n)
            total=0
            for th in thetas:
                rvals=np.linspace(rl(th),ru(th),n)
                integrand=f(rvals,th)*rvals
                total+=np.trapz(integrand,rvals)
            total*=((t2-t1)/n)
            st.success(total)
        except Exception as e: st.error(e)

# --- Triple Integral ---
if tool == "Triple Integral with 3D Graph":
    st.header("🔺 Triple Integral + 3D Plot")
    x,y,z=sp.symbols("x y z")
    f_str=st.text_input("f(x,y,z):","x*y*z")
    xl=st.text_input("x min:","0")
    xu=st.text_input("x max:","1")
    yl=st.text_input("y min:","0")
    yu=st.text_input("y max:","1")
    zl=st.text_input("z min:","0")
    zu=st.text_input("z max:","1")
    if st.button("Calculate"):
        try:
            f=sp.sympify(f_str)
            res=sp.integrate(f,(z,sp.sympify(zl),sp.sympify(zu)),
                               (y,sp.sympify(yl),sp.sympify(yu)),
                               (x,sp.sympify(xl),sp.sympify(xu)))
            st.success(res)
            X=[float(eval(xl)),float(eval(xu))]
            Y=[float(eval(yl)),float(eval(yu))]
            Z=[float(eval(zl)),float(eval(zu))]
            fig=go.Figure()
            fig.add_trace(go.Mesh3d(
                x=[X[0],X[1],X[1],X[0],X[0],X[1],X[1],X[0]],
                y=[Y[0],Y[0],Y[1],Y[1],Y[0],Y[0],Y[1],Y[1]],
                z=[Z[0],Z[0],Z[0],Z[0],Z[1],Z[1],Z[1],Z[1]],
                opacity=0.5,color="lightblue"))
            st.plotly_chart(fig,use_container_width=True)
        except Exception as e: st.error(e)
