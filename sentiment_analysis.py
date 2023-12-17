import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_name='roberta-base', classifier_model='arpanghoshal/EmoRoBERTa'):
        """
        Initializes the sentiment analyzer with the specified models.
        :param model_name: Name of the tokenizer model
        :param classifier_model: Name of the sentiment classification model
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = TFRobertaForSequenceClassification.from_pretrained(classifier_model)

    def analyze_sentiment(self, user_input):
        """
        Analyzes the sentiment of the given user input.
        :param user_input: Text input from the user
        :return: A tuple of sentiment label and sentiment score
        """
        encoded_input = self.tokenizer(user_input, return_tensors="tf", truncation=True, padding=True, max_length=512)
        outputs = self.model(encoded_input)
        scores = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        predicted_class_idx = tf.argmax(outputs.logits, axis=-1).numpy()[0]
        sentiment_label = self.model.config.id2label[predicted_class_idx]
        sentiment_score = scores[predicted_class_idx]
        return sentiment_label, sentiment_score
