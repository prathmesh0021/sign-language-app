from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(prediction, confidence, top5):

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Sign Language Recognition Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence}", styles['Normal']))
    content.append(Spacer(1, 20))

    content.append(Paragraph("Top-5 Predictions:", styles['Heading2']))
    for i, item in enumerate(top5):
        content.append(Paragraph(f"{i+1}. {item}", styles['Normal']))

    content.append(Spacer(1, 20))

    content.append(Paragraph("Model Performance Graphs", styles['Heading2']))

    graphs = [
        "static/graphs/confusion_matrix.png",
        "static/graphs/metrics.png",
        "static/graphs/test_accuracy.png"
    ]

    for g in graphs:
        try:
            content.append(Image(g, width=400, height=250))
            content.append(Spacer(1, 10))
        except:
            pass

    doc.build(content)

    return "report.pdf"