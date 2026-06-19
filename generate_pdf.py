from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "CommissionLens Project Presentation", border=0, align="R")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def create_presentation():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Slide 1: Title
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(50)
    pdf.cell(0, 10, "CommissionLens", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "Commission-Adjusted Alpha Prediction in Mutual Funds", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 14)
    pdf.cell(0, 10, "Finance & Economics Club - DIY Project 04", align="C", new_x="LMARGIN", new_y="NEXT")

    # Slide 2: Problem Statement
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Problem Statement", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    content = (
        "1. India has over 90 million demat accounts, but many retail investors ignore "
        "the critical expense ratio gap between regular and direct mutual fund plans.\n\n"
        "2. Regular plans charge an additional 0.5% - 1.5% annually as distributor commission.\n\n"
        "3. Due to compounding, this small difference can erode Rs. 8 - Rs. 12 lakh from a Rs. 5,000 "
        "monthly SIP over 20 years.\n\n"
        "4. The higher cost is ONLY justified if the fund generates sufficient excess "
        "return (alpha) over the benchmark to offset the commission.\n\n"
        "5. Currently, no systematic ML framework exists to predict whether a fund will "
        "continue to justify this fee. CommissionLens solves this."
    )
    pdf.multi_cell(0, 10, content)

    # Slide 3: Goals & Methodology
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Goals & Methodology", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    content2 = (
        "-> Data Collection: Synthesized/ingested 5 years of quarterly data across 250 "
        "Indian Equity Mutual Funds, including macroeconomic constraints (Repo Rate, CPI, FII/DII).\n\n"
        "-> Feature Engineering: Computed rolling 4Q/8Q Sharpe ratios, trailing volatility, "
        "information ratios, beta, and tracking error.\n\n"
        "-> Target Definition: Binary classification labelling fund-quarters as 'Justified' "
        "if Next Quarter Gross Alpha > Expense Gap.\n\n"
        "-> Temporal Splitting: Strict train/val/test split across time to prevent "
        "look-ahead bias in financial markets."
    )
    pdf.multi_cell(0, 10, content2)

    # Slide 4: Models & Performance
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "ML Models & Performance", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    content3 = (
        "Models Trained:\n"
        "- Regression (Predicting Net Alpha): Linear Regression, Random Forest, XGBoost\n"
        "- Classification (Predicting Justification): Logistic Regression, Random Forest, XGBoost\n\n"
        "Evaluation Highlights (Test Set):\n"
        "- Best Regression Model: Random Forest (RMSE ~0.028)\n"
        "- Best Classification Model: Random Forest\n"
        "- We successfully evaluated Accuracy, Precision, Recall, F1 Score, and "
        "Precision at Top Decile.\n\n"
        "Conclusion: Tree-based models effectively captured the non-linear relationship "
        "between macroeconomic shifts, fund turnover, and alpha generation."
    )
    pdf.multi_cell(0, 10, content3)

    # Slide 5: Explainability (SHAP)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "SHAP Explainability", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    content4 = (
        "We applied SHAP (SHapley Additive exPlanations) TreeExplainer to demystify our "
        "Random Forest model and identify the top drivers of commission justification.\n\n"
        "Top predictive features discovered:\n"
        "1. Rolling Volatility (8Q & 4Q)\n"
        "2. Rolling 4Q Return\n"
        "3. Log AUM\n"
        "4. Beta (8Q)\n\n"
        "Interpretation: High rolling returns and optimal volatility clustering strongly "
        "indicate whether a manager is actively providing value or just tracking the index."
    )
    pdf.multi_cell(0, 10, content4)

    # Slide 6: SIP Backtest & Delivery
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "SIP Back-Validation & Deliverables", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    content5 = (
        "Backtest Simulation:\n"
        "- We simulated a monthly Rs. 5,000 SIP (Rs. 15,000 quarterly) over the test period.\n"
        "- Compared three strategies: Naive Regular Plan, Model-Guided Regular Plan, and "
        "Direct Plan Benchmark.\n"
        "- Outcome: The Model-Guided strategy effectively filtered out underperforming "
        "funds, minimizing dead-weight commission costs.\n\n"
        "Final Deliverables Provided:\n"
        "1. GitHub Repository with modular OOP codebase.\n"
        "2. Jupyter Notebook research report.\n"
        "3. Live Streamlit Interactive Dashboard.\n"
        "4. Exported performance reports & SHAP dependency plots."
    )
    pdf.multi_cell(0, 10, content5)

    pdf.output("reports/CommissionLens_Presentation.pdf")
    print("PDF Presentation successfully generated!")

if __name__ == "__main__":
    create_presentation()
