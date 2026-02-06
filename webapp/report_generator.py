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

    def add_anomaly(self, index, log_details, score, explanation=None, is_critical=False):
        if is_critical:
            self.set_text_color(220, 53, 69) # Red
            title = f"CRITICAL Anomaly #{index} (Score/Error: {score:.4f})"
        else:
            self.set_text_color(0, 0, 0) # Black
            title = f"Anomaly #{index} (Score/Error: {score:.4f})"
            
        self.set_font('Arial', 'B', 10)
        self.cell(0, 6, title, 0, 1)
        
        self.set_text_color(0, 0, 0) # Reset to black
        self.set_font('Courier', '', 8)
        self.multi_cell(0, 4, f"Request: {log_details.get('request', 'N/A')}")
        self.multi_cell(0, 4, f"Status: {log_details.get('status', 'N/A')} | Bytes: {log_details.get('bytes', 'N/A')} | IP: {log_details.get('ip1', 'N/A')}")
        
        if explanation:
            self.ln(2)
            self.set_font('Arial', 'B', 9)
            self.multi_cell(0, 5, f"Reason: {explanation}")
        
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
            item.get('explanation'),
            item.get('is_critical', False)
        )
        
    output_path = os.path.join(os.path.dirname(__file__), 'uploads', f'report_{model_name}_{filename}.pdf')
    pdf.output(output_path, 'F')
    return output_path
