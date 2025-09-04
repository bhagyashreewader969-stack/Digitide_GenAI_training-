import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title='LoanLab — Advanced Loan Calculator', page_icon='💸', layout='wide')

# CSS and small hover effects
st.markdown('''
<style>
.card {transition: all 0.25s ease; padding: 12px; border-radius: 12px; background: var(--secondary-background-color);}
.card:hover {transform: translateY(-6px); box-shadow: 0 8px 20px rgba(0,0,0,0.12);}
.stButton>button:hover {transform: scale(1.03);}
[data-testid="stMetric"] {transition: transform 0.15s ease;}
[data-testid="stMetric"]:hover {transform: scale(1.05);}
.milestone {padding:6px 10px; border-radius:8px; display:inline-block; margin-right:8px;}
</style>
''', unsafe_allow_html=True)

if 'dark' not in st.session_state:
    st.session_state.dark = False
col_top = st.columns([3,1])
with col_top[1]:
    if st.button('Toggle Dark Mode 🌙'):
        st.session_state.dark = not st.session_state.dark
if st.session_state.dark:
    st.markdown("<style>body {background-color:#0f1724; color:white}</style>", unsafe_allow_html=True)

# Helpers
def payment_amount(P, r_annual, n_years, m_payments):
    n = int(round(n_years * m_payments))
    if n == 0: return 0.0, 0
    r = (r_annual/100.0)/m_payments
    if r == 0: return P/n, n
    A = P * (r * (1+r)**n) / ((1+r)**n - 1)
    return A, n

def amort_schedule(P, r_annual, n_years, m_payments, extra=0.0, prepay=None, variable_rates=None):
    rows=[]; balance=P; k=0; total_interest=0.0
    rates = None
    if variable_rates:
        rates=[] 
        for periods_count, rate in variable_rates:
            rates += [rate]*periods_count
    max_iter=200000
    while balance>1e-6 and k<max_iter:
        k+=1
        if rates:
            r_annual = rates[k-1] if k-1<len(rates) else rates[-1]
        r = (r_annual/100.0)/m_payments
        A, _ = payment_amount(balance, r_annual, 1.0, m_payments)
        interest = balance * r
        principal = min(balance, A - interest + extra)
        balance = max(0.0, balance - principal)
        prepay_now=0.0
        if prepay and k==prepay[0]:
            prepay_now = min(balance, prepay[1]); balance = max(0.0, balance - prepay_now)
        total_interest += interest
        rows.append({'Payment #':k,'Payment':round(A,2),'Extra':round(extra,2),'Prepayment':round(prepay_now,2),'Interest':round(interest,2),'Principal':round(principal+prepay_now,2),'Remaining Balance':round(balance,2)})
        if k>100000: break
    return pd.DataFrame(rows), total_interest

def encode_inputs_to_url(params):
    import urllib.parse
    return urllib.parse.urlencode(params)

def calc_npv(cashflows, rate):
    return sum([cf/((1+rate)**i) for i,cf in enumerate(cashflows)])

def calc_irr(cashflows):
    try:
        r=0.1
        for _ in range(100):
            f=sum([cf/((1+r)**i) for i,cf in enumerate(cashflows)])
            df=sum([-i*cf/((1+r)**(i+1)) for i,cf in enumerate(cashflows)])
            if df==0: break
            r_new=r - f/df
            if abs(r_new-r)<1e-6: return r_new
            r=r_new
        return r
    except:
        return None

# Layout
st.title('LoanLab — Advanced Loan Calculator 💸')
st.sidebar.header('Global Settings')
payments_per_year = st.sidebar.selectbox('Payments per year',[12,24,26,52], index=0)
income = st.sidebar.number_input('Monthly Income (for risk alerts)', min_value=0.0, value=50000.0, step=1000.0)

num_loans = st.sidebar.slider('Number of loans to compare',1,3,1)
loans=[]
for i in range(num_loans):
    with st.sidebar.expander(f'Loan {i+1} inputs', expanded=(i==0)):
        name=st.text_input(f'Loan {i+1} - Name', value=f'Loan {i+1}', key=f'n{i}')
        price=st.number_input(f'Loan {i+1} - Price', min_value=0.0, value=1000000.0, step=50000.0, key=f'pr{i}')
        deposit=st.number_input(f'Loan {i+1} - Deposit', min_value=0.0, value=200000.0, step=10000.0, key=f'd{i}')
        principal=max(0.0, price-deposit)
        rate=st.number_input(f'Loan {i+1} - Annual Rate %', min_value=0.0, value=8.5, step=0.1, key=f'r{i}')
        years=st.number_input(f'Loan {i+1} - Years', min_value=0.5, value=20.0, step=0.5, key=f'y{i}')
        extra=st.number_input(f'Loan {i+1} - Extra per payment', min_value=0.0, value=0.0, step=500.0, key=f'e{i}')
        prepay_toggle=st.checkbox(f'Loan {i+1} - One-time prepayment', value=False, key=f'pt{i}')
        if prepay_toggle:
            prepay_period=st.number_input(f'Loan {i+1} - Prepay at payment #', min_value=1, value=12, key=f'pp{i}')
            prepay_amount=st.number_input(f'Loan {i+1} - Prepay amount', min_value=0.0, value=0.0, key=f'pa{i}')
            prepay=(int(prepay_period), float(prepay_amount))
        else:
            prepay=None
        var_toggle=st.checkbox(f'Loan {i+1} - Variable rate?', value=False, key=f'vt{i}')
        variable_rates=None
        if var_toggle:
            st.markdown('Enter variable rate segments as CSV: periods,annual_rate (e.g. `60,6.5\n180,8.0`)')
            txt=st.text_area(f'Loan {i+1} - Variable rate CSV', value='60,6.5\n180,8.0', key=f'vr{i}', height=80)
            try:
                lines=[ln.strip() for ln in txt.splitlines() if ln.strip()]
                segs=[]
                for ln in lines:
                    p,r=ln.split(','); segs.append((int(p.strip()), float(r.strip())))
                variable_rates=segs
            except:
                st.error('Variable rate CSV invalid.')
        loans.append({'name':name,'price':price,'deposit':deposit,'principal':principal,'rate':rate,'years':years,'extra':extra,'prepay':prepay,'variable_rates':variable_rates})

# Main
st.header('Comparison & Analysis')
cols=st.columns([1,1,1])
for idx, loan in enumerate(loans):
    df, tot_interest = amort_schedule(loan['principal'], loan['rate'], loan['years'], payments_per_year, extra=loan['extra'], prepay=loan['prepay'], variable_rates=loan['variable_rates'])
    loan['df']=df; loan['total_interest']=tot_interest; loan['periods']=len(df)
    payment = df['Payment'].iloc[0] if not df.empty else 0.0
    monthly_equiv = payment * (12/payments_per_year)
    with cols[idx]:
        st.markdown(f"<div class='card'><h4>{loan['name']}</h4><p>Principal: ₹{loan['principal']:,.0f}</p></div>", unsafe_allow_html=True)
        st.metric('Payment per period', f'₹{payment:,.2f}'); st.metric('Est. Monthly Equivalent', f'₹{monthly_equiv:,.2f}'); st.metric('Total Interest', f'₹{tot_interest:,.2f}')

# Balance comparison
st.markdown('### Balance over time (comparison)')
plt.figure(figsize=(8,3))
for loan in loans:
    if not loan['df'].empty:
        plt.plot(loan['df']['Payment #'], loan['df']['Remaining Balance'], label=loan['name'])
plt.xlabel('Payment #'); plt.ylabel('Remaining Balance (₹)'); plt.legend(); st.pyplot(plt.gcf()); plt.clf()

# Pie chart for first loan
st.markdown('### EMI Breakdown — First Loan')
if loans:
    first=loans[0]; df=first['df']
    if not df.empty:
        total_principal = df['Principal'].sum(); total_interest = df['Interest'].sum()
        fig1, ax1 = plt.subplots(); ax1.pie([total_principal, total_interest], labels=['Principal','Interest'], autopct='%1.1f%%', startangle=140); ax1.set_title('EMI: Principal vs Interest'); st.pyplot(fig1)

# Progress & milestones for first loan (demonstration)
st.markdown('### Loan Journey & Progress')
if loans:
    df=loans[0]['df']
    if not df.empty:
        payments_total = loans[0]['periods']; payments_done = len(df)
        progress = min(1.0, payments_done/payments_total) if payments_total>0 else 0
        st.progress(int(progress*100))
        milestones=[0.25,0.5,0.75,1.0]; cols_ms = st.columns(len(milestones))
        for i,m in enumerate(milestones):
            achieved = progress>=m
            text = f"<span class='milestone' style='background:{'#0f0' if achieved else '#444'}'>{int(m*100)}% {'🎉' if achieved else ''}</span>"
            cols_ms[i].markdown(text, unsafe_allow_html=True)

# Analytics: NPV & IRR for first loan
st.markdown('### Analytics (First Loan)')
if loans:
    df=loans[0]['df']
    if not df.empty:
        discount = st.number_input('Discount rate for NPV (%)', min_value=0.0, value=6.0, step=0.1)
        cashflows = [-loans[0]['principal']] + list(-(df['Payment']+df['Extra']+df['Prepayment']).round(2).astype(float))
        npv = calc_npv(cashflows, discount/100.0); irr = calc_irr(cashflows)
        st.write(f"NPV @ {discount}% = ₹{npv:,.2f}"); 
        if irr is not None: st.write(f"Approx IRR = {irr*100:.2f}%") 
        else: st.write('IRR could not be computed.')

# Stress test +2%
st.markdown('### Stress Test: +2% interest')
for loan in loans:
    df2, ti2 = amort_schedule(loan['principal'], loan['rate']+2.0, loan['years'], payments_per_year, extra=loan['extra'], prepay=loan['prepay'], variable_rates=loan['variable_rates'])
    st.write(f"{loan['name']}: total interest now ₹{ti2:,.2f} (was ₹{loan.get('total_interest',0):,.2f})")

# Risk alerts
st.markdown('### Risk Alerts')
for loan in loans:
    df = loan['df']; payment = df['Payment'].iloc[0] if not df.empty else 0.0; monthly_equiv = payment * (12/payments_per_year)
    if monthly_equiv > 0.4*income: st.warning(f"{loan['name']}: Estimated monthly equivalent ₹{monthly_equiv:,.0f} is >40% of income — high risk.")
    elif monthly_equiv > 0.3*income: st.info(f"{loan['name']}: Estimated monthly equivalent ₹{monthly_equiv:,.0f} is >30% of income — caution.")

# Export Excel & PDF
st.markdown('### Export & Share')
if st.button('Download Excel (first loan)') and loans:
    df = loans[0]['df']
    if not df.empty:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Amortization')
        buf.seek(0)
        st.download_button('⬇️ Download XLSX', data=buf, file_name='amortization.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if st.button('Download PDF Summary (first loan)') and loans:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    df = loans[0]['df']
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6,3)); ax.plot(df['Payment #'], df['Remaining Balance']); ax.set_title('Remaining Balance'); buf=io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
        pdfbuf = io.BytesIO(); c = canvas.Canvas(pdfbuf, pagesize=A4); c.drawString(40,800, f"Loan Summary — {loans[0]['name']}"); from reportlab.lib.utils import ImageReader; img=ImageReader(buf); c.drawImage(img,40,450,width=500,height=250); c.showPage(); c.save(); pdfbuf.seek(0); st.download_button('⬇️ Download PDF', data=pdfbuf, file_name='loan_summary.pdf', mime='application/pdf')

# Shareable link
if st.button('Generate shareable query string'):
    params={'num_loans':num_loans}
    for i,loan in enumerate(loans):
        params[f'name{i}']=loan['name']; params[f'price{i}']=loan['price']; params[f'deposit{i}']=loan['deposit']; params[f'rate{i}']=loan['rate']
    q=encode_inputs_to_url(params); st.code(q)

st.markdown('\n---\nMade with ❤️ — LoanLab')
