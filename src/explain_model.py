import shap
import lime.lime_text

def explain_with_shap(model, vectorizer, texts):
    explainer = shap.Explainer(model, vectorizer.transform)
    shap_values = explainer(texts[:5])
    shap.plots.text(shap_values[0])

def explain_with_lime(model, vectorizer, class_names, text_sample):
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    pipeline = lambda x: model.predict_proba(vectorizer.transform(x))
    exp = explainer.explain_instance(text_sample, pipeline, num_features=6)
    exp.show_in_notebook()
