from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class DocumentClassifier:
    """Local, lightweight Machine Learning model to categorize corporate documents."""
    
    def __init__(self):
        # Build a robust textual pipeline: Text -> Mathematical Vectors -> Prediction
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
            ('clf', MultinomialNB())
        ])
        self._train_local_model()

    def _train_local_model(self):
        """Trains the model instantly on boot using localized SME templates."""
        # Synthetic training data representing typical SME business documents (6 items total)
        training_texts = [
            # Invoices (Samples 1 & 2)
            "INVOICE Bill To: Customer Amount Due KSH 50,000 Supply of office equipment Tax Invoice Bank Details Balance",
            "Invoice invoice payment requested receipt total due bank account number branch swift code billing",
            
            # Employment Contracts (Samples 3 & 4)
            "EMPLOYMENT CONTRACT This Agreement is made between the Employer and the Employee Job Title Salary Leave Benefits Termination",
            "Employment Agreement probationary period permanent employment position description basic salary allowances ksh",
            
            # National IDs (Samples 5 & 6)
            "REPUBLIC OF KENYA NATIONAL IDENTITY CARD ID NUMBER ID NO Names Date of Birth Sex Signature copy Holder",
            "KENYA NATIONAL ID CARD Identity Document Date of Issue District of Birth Serial Number card holder copy"
        ]
        
        # Corresponding targets aligned perfectly (6 items total)
        training_labels = [
            "Invoice", "Invoice",
            "Contract", "Contract",
            "National ID", "National ID"
        ]
        
        # Trigger training fit
        self.model.fit(training_texts, training_labels)
        print("[+] Phase 2: Local document classification model trained successfully.")

    def predict_type(self, extracted_text: str) -> str:
        """Analyzes raw string variables in RAM and outputs the predicted document type."""
        if not extracted_text.strip():
            return "Unknown Structure"
        
        prediction = self.model.predict([extracted_text])
        return prediction[0]

if __name__ == "__main__":
    classifier = DocumentClassifier()
    test_text = "Tax Invoice details: Total due is KSH 12,000 payable to account standard bank."
    print(f"[*] Test Prediction Result: {classifier.predict_type(test_text)}")