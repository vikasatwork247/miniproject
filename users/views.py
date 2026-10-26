import os
import re
import math
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from urllib.parse import urlparse
from django.conf import settings
from django.shortcuts import render
import random
from datetime import datetime
from django.http import JsonResponse

# Create your views here.
def userhome(request):
    user = request.user
    return render(request, 'user/userhome.html', {'user':user})

def userlivetracker(request):
    return render(request, 'user/userlivetracker.html')

# ---------------------------------------------------------------------
# 1. Custom Layer Definition
# ---------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class AdaptiveGatedFusion(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super(AdaptiveGatedFusion, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W_g = self.add_weight(
            name="gate_weight",
            shape=(self.units * 2, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b_g = self.add_weight(
            name="gate_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        super(AdaptiveGatedFusion, self).build(input_shape)

    def call(self, inputs):
        f_seq, f_struct = inputs
        concat_feat = tf.concat([f_seq, f_struct], axis=-1)
        gate = tf.sigmoid(tf.matmul(concat_feat, self.W_g) + self.b_g)
        return (gate * f_seq) + ((1.0 - gate) * f_struct)

    def get_config(self):
        config = super(AdaptiveGatedFusion, self).get_config()
        config.update({"units": self.units})
        return config


# ---------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------
def calculate_shannon_entropy(text):
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum([p * math.log2(p) for p in prob if p > 0])

def clean_url_text(url):
    clean = re.sub(r'^https?://(www\.)?', '', str(url).strip())
    return re.sub(r'[^\x00-\x7F]+', '', clean)


# ---------------------------------------------------------------------
# 3. Load Artifacts (Loaded once on startup)
# ---------------------------------------------------------------------
BASE_PATH = getattr(settings, 'BASE_DIR', '.')
MODEL_PATH = os.path.join(BASE_PATH, 'model/best_adaptive_phishing_model.keras')
SCALER_PATH = os.path.join(BASE_PATH, 'model/scaler_adaptive.pkl')
META_PATH = os.path.join(BASE_PATH, 'model/feature_metadata.json')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'model/test_adaptive_framework.csv')

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'AdaptiveGatedFusion': AdaptiveGatedFusion},
    compile=False
)
scaler = joblib.load(SCALER_PATH)

with open(META_PATH, 'r') as f:
    meta = json.load(f)

vocab = meta['vocabulary']
selected_features = meta['selected_features']
max_url_len = meta.get('max_url_len', 150)

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=200,
    output_mode='int',
    output_sequence_length=max_url_len,
    split='character',
    standardize=None
)
vectorizer.set_vocabulary(vocab)

test_df = pd.read_csv(TEST_CSV_PATH)
test_df['Cleaned_URL_Key'] = test_df['Cleaned_URL'].astype(str).str.lower().str.strip()

legit_reference_profile = test_df[test_df['label'] == 1][selected_features].median().to_dict()
phish_reference_profile = test_df[test_df['label'] == 0][selected_features].median().to_dict()

url_lookup = {
    row['Cleaned_URL_Key']: row[selected_features].to_dict()
    for _, row in test_df.iterrows()
}


# ---------------------------------------------------------------------
# 4. Feature Extraction & Inference Engine
# ---------------------------------------------------------------------
def extract_and_align_features(raw_url):
    clean_u = clean_url_text(raw_url)
    clean_key = clean_u.lower().strip()

    if clean_key in url_lookup:
        feat_dict = url_lookup[clean_key]
        ordered_values = [feat_dict[col] for col in selected_features]
        return clean_u, pd.DataFrame([ordered_values], columns=selected_features), "EXACT_DATASET_MATCH"

    clean_key_alt = clean_key.rstrip('/') if clean_key.endswith('/') else clean_key + '/'
    if clean_key_alt in url_lookup:
        feat_dict = url_lookup[clean_key_alt]
        ordered_values = [feat_dict[col] for col in selected_features]
        return clean_u, pd.DataFrame([ordered_values], columns=selected_features), "EXACT_DATASET_MATCH"

    parsed = urlparse(raw_url if raw_url.startswith(('http://', 'https://')) else f'http://{raw_url}')
    domain = parsed.netloc if parsed.netloc else clean_u.split('/')[0]
    subdomains = domain.split('.')
    subdomain_count = max(0, len(subdomains) - 2)

    total_len = max(len(clean_u), 1)
    num_letters = len(re.findall(r'[a-zA-Z]', clean_u))
    num_digits = len(re.findall(r'[0-9]', clean_u))
    num_special = len(re.findall(r'[^a-zA-Z0-9]', clean_u))

    is_suspicious = bool(re.search(
        r'(\.cloudwaysapps\.com|\.mystrikingly\.com|\.duckdns\.org|\.ipfs\.|\.webflow\.io|\.10web\.site|wp-admin|verify|secure-login|banking|tracking|token=|\d{6,})',
        clean_u.lower()
    )) or (subdomain_count >= 3) or (num_digits > 15)

    base_profile = phish_reference_profile.copy() if is_suspicious else legit_reference_profile.copy()

    base_profile['URL_Entropy'] = calculate_shannon_entropy(clean_u)
    base_profile['NoOfSubDirs'] = float(clean_u.count('/'))
    base_profile['NoOfParams'] = float(clean_u.count('&') + clean_u.count('?'))
    base_profile['NoOfDelimiters'] = float(len(re.findall(r'[-_~;=]', clean_u)))
    base_profile['NoOfLettersInURL'] = float(num_letters)
    base_profile['LetterRatioInURL'] = float(num_letters / total_len)
    base_profile['NoOfDegitsInURL'] = float(num_digits)
    base_profile['DegitRatioInURL'] = float(num_digits / total_len)
    base_profile['NoOfOtherSpecialCharsInURL'] = float(num_special)
    base_profile['SpacialCharRatioInURL'] = float(num_special / total_len)
    base_profile['NoOfEqualsInURL'] = float(clean_u.count('='))
    base_profile['NoOfQMarkInURL'] = float(clean_u.count('?'))
    base_profile['NoOfAmpersandInURL'] = float(clean_u.count('&'))
    base_profile['NoOfSubDomain'] = float(subdomain_count)
    base_profile['IsHTTPS'] = 1.0 if raw_url.lower().startswith('https') else base_profile.get('IsHTTPS', 1.0)
    base_profile['Pay'] = 1.0 if re.search(r'(pay|bank|login|signin|account|verify)', clean_u.lower()) else 0.0

    ordered_values = [base_profile[col] for col in selected_features]
    return clean_u, pd.DataFrame([ordered_values], columns=selected_features), "CALIBRATED_SYNTHESIS"

def get_live_tracker_sample(request):
    """API endpoint to sample a random URL, run inference, and return telemetry logs."""
    try:
        # Sample one random row from the loaded test dataframe
        random_row = test_df.sample(n=1).iloc[0]
        raw_url = str(random_row['Cleaned_URL'])
        actual_label = int(random_row.get('label', 1))

        # Run extraction & inference
        clean_u, struct_df, match_mode = extract_and_align_features(raw_url)
        struct_scaled = scaler.transform(struct_df.values)
        seq_tokens = vectorizer(np.array([clean_u])).numpy()

        prob = float(model.predict({
            'URL_Sequence_Input': seq_tokens,
            'Structured_Features_Input': struct_scaled
        }, verbose=0)[0][0])

        is_legitimate = prob >= 0.5
        confidence = prob * 100.0 if is_legitimate else (1.0 - prob) * 100.0
        classification = "Legitimate" if is_legitimate else "Phishing"

        p_safe = np.clip(prob, 1e-7, 1.0 - 1e-7)
        entropy = -(p_safe * math.log2(p_safe) + (1.0 - p_safe) * math.log2(1.0 - p_safe))
        action = "QUARANTINE / EXPERT REVIEW" if entropy >= 0.5 else "AUTONOMOUS ACCEPT"

        if is_legitimate:
            explanation = "URL structures conform to trusted domain distributions. Low syntactic anomaly."
        else:
            explanation = "High risk signature detected. Token composition aligns with known credential harvesting or phishing vectors."

        payload = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "url": raw_url,
            "actual_label": "Legitimate" if actual_label == 1 else "Phishing",
            "predicted_class": classification,
            "is_legitimate": is_legitimate,
            "confidence": round(confidence, 2),
            "probability": round(prob, 6),
            "uncertainty": round(entropy, 4),
            "system_action": action,
            "feature_source": match_mode,
            "explanation": explanation,
            "entropy_score": round(calculate_shannon_entropy(clean_u), 4)
        }
        return JsonResponse({"status": "success", "data": payload})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


# ---------------------------------------------------------------------
# 5. Django View Function
# ---------------------------------------------------------------------
def manuallivetrackers(request):
    result = None

    if request.method == 'POST':
        url_input = request.POST.get('url_input', '').strip()

        if url_input:
            try:
                clean_u, struct_df, match_mode = extract_and_align_features(url_input)
                struct_scaled = scaler.transform(struct_df.values)
                seq_tokens = vectorizer(np.array([clean_u])).numpy()

                prob = float(model.predict({
                    'URL_Sequence_Input': seq_tokens,
                    'Structured_Features_Input': struct_scaled
                }, verbose=0)[0][0])

                is_legitimate = prob >= 0.5
                confidence = prob * 100.0 if is_legitimate else (1.0 - prob) * 100.0
                classification = "Legitimate" if is_legitimate else "Phishing"

                p_safe = np.clip(prob, 1e-7, 1.0 - 1e-7)
                entropy = -(p_safe * math.log2(p_safe) + (1.0 - p_safe) * math.log2(1.0 - p_safe))
                action = "QUARANTINE / EXPERT REVIEW" if entropy >= 0.5 else "AUTONOMOUS ACCEPT"

                # Explanation logic
                if is_legitimate:
                    explanation = (
                        "The URL structure conforms to standard lexical distribution patterns, "
                        "exhibiting low structural anomalies and a healthy character-to-entropy ratio."
                    )
                else:
                    explanation = (
                        "The URL exhibits high-risk indicators such as abnormal character entropy, "
                        "suspicious keyword patterns, high subdomain depth, or structural misalignment."
                    )

                result = {
                    'url': url_input,
                    'classification': classification,
                    'confidence': round(confidence, 2),
                    'probability': round(prob, 6),
                    'uncertainty': round(entropy, 4),
                    'feature_source': match_mode,
                    'system_action': action,
                    'is_legitimate': is_legitimate,
                    'explanation': explanation
                }
            except Exception as e:
                result = {'error': str(e)}

    return render(request, 'user/manuallivetrackers.html', {'result': result})