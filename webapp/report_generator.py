from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Anomaly Detection Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_anomaly(self, index, log_details, score, explanation=None, attack_type=None, severity="Low", llm_prediction=None, is_critical=False):
        # Title Color based on Severity
        if severity == "Critical":
            self.set_text_color(220, 53, 69) # Red
        elif severity == "High":
             self.set_text_color(255, 193, 7) # Orange/Yellow (Darker for text)
        elif severity == "Medium":
            self.set_text_color(23, 162, 184) # Teal
        else:
            self.set_text_color(0, 0, 0) # Black

        title = f"Anomaly #{index} | Severity: {severity} | Type: {attack_type if attack_type else 'Unknown'}"
        title += f" (Score: {score:.4f})"
            
        self.set_font('Arial', 'B', 10)
        self.cell(0, 6, title, 0, 1)
        
        self.set_text_color(0, 0, 0) # Reset to black
        
        # Print Full Log Details
        self.set_font('Courier', '', 8)
        self.cell(0, 5, "Log Details:", 0, 1)
        log_str = ""
        for k, v in log_details.items():
            log_str += f"{k}: {v}  "
        self.multi_cell(0, 4, log_str)
        
        # LLM Prediction
        if llm_prediction:
            self.ln(2)
            self.set_font('Arial', 'B', 9)
            self.cell(0, 5, "LLM Analysis:", 0, 1)
            self.set_font('Arial', '', 9)
            self.multi_cell(0, 5, llm_prediction)

        # AI Explanation
        if explanation:
            self.ln(2)
            self.set_font('Arial', 'B', 9)
            self.cell(0, 5, "Reason for Detection:", 0, 1)
            self.set_font('Arial', 'I', 9)
            self.multi_cell(0, 5, explanation)
        
        self.ln(4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

def generate_report(model_name, anomalies, filename):
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Model: {model_name}", 0, 1)
    pdf.cell(0, 10, f"Source File: {filename}", 0, 1)
    pdf.cell(0, 10, f"Total Anomalies Found: {len(anomalies)}", 0, 1)
    pdf.ln(10)
    
    for item in anomalies:
        pdf.add_anomaly(
            item.get('index', 0), 
            item.get('log', {}), 
            item.get('error') if 'error' in item else item.get('score', 0),
            explanation=item.get('explanation'),
            attack_type=item.get('attack_type'),
            severity=item.get('severity'),
            llm_prediction=item.get('llm_prediction'),
            is_critical=item.get('is_critical', False)
        )
        
    output_path = os.path.join(os.path.dirname(__file__), 'uploads', f'report_{model_name}_{filename}.pdf')
    pdf.output(output_path, 'F')
    return output_path
