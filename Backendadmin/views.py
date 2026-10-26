from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

import io
import base64
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

def index(request):
    return render(request, "index.html")

def loginn(request):
    return render(request, "login.html")

def register(request):
    return render(request, "register.html")

# Define the login function
def user_login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            if not user.is_active:
                # User is inactive
                messages.error(request, "Your account is inactive. Please contact the admin.")
                return redirect('loginn')

            # Login the user
            login(request, user)

            if user.is_staff or user.is_superuser:
                # Redirect to admin home if user is staff
                return redirect('adminhome')
            else:
                # Redirect to user home if user is not staff
                return redirect('userhome')
        else:
            # Invalid username or password
            messages.error(request, "Invalid username or password.")
            return redirect('loginn')

    return render(request, 'login.html')

# Define the user registration function
def user_registration(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')

        # Check if passwords match
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('register')

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('register')

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('register')

        # Create the user with is_active set to False
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )
        user.save()

        messages.success(request, "Registration successful! Please wait for admin approval.")
        return redirect('loginn')

    return render(request, 'register.html')

# Define the logout function
def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('index')

def adminhome(request):
    users = User.objects.filter(is_staff=False, is_superuser=False) 
    return render(request, "admin/adminhome.html", {"users_list": users})

def admin_update_userstatus(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        
        # Toggle the is_active status
        user.is_active = not user.is_active
        user.save()

        # Display message based on the action
        if user.is_active:
            messages.success(request, f"User {user.username} has been activated.")
        else:
            messages.success(request, f"User {user.username} has been deactivated.")
        
        return redirect('adminhome')  # Redirect back to the admin home page
    except User.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('adminhome')

def modeltraining(request):
    report_data = None
    error_message = None

    if request.method == 'POST' and request.FILES.get('training_csv'):
        try:
            from users.views import extract_and_align_features, selected_features, scaler, vectorizer, model

            csv_file = request.FILES['training_csv']
            df = pd.read_csv(csv_file)

            # Standardize URL and label column lookups
            url_col = next((c for c in df.columns if c.lower() in ['cleaned_url', 'url', 'urls']), None)
            label_col = next((c for c in df.columns if c.lower() in ['label', 'target', 'class', 'labels']), None)

            if not url_col or not label_col:
                raise ValueError("Uploaded CSV must contain a URL column ('Cleaned_URL' or 'URL') and a Label column ('label' or 'class').")

            # Extract features for all entries in the uploaded file
            clean_urls = []
            struct_features_list = []
            y_true = df[label_col].astype(int).values

            for _, row in df.iterrows():
                raw_u = str(row[url_col])
                clean_u, struct_row_df, _ = extract_and_align_features(raw_u)
                clean_urls.append(clean_u)
                struct_features_list.append(struct_row_df.iloc[0])

            struct_matrix = pd.DataFrame(struct_features_list)[selected_features]
            struct_scaled = scaler.transform(struct_matrix.values)
            seq_tokens = vectorizer(np.array(clean_urls)).numpy()

            # Batch Inference
            y_probs = model.predict({
                'URL_Sequence_Input': seq_tokens,
                'Structured_Features_Input': struct_scaled
            }, batch_size=64, verbose=0).flatten()

            y_pred = (y_probs >= 0.5).astype(int)

            # Compute standard evaluation metrics
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)

            # Format detailed per-class report with Django template-safe keys
            cls_report_raw = classification_report(
                y_true,
                y_pred,
                target_names=['Phishing (0)', 'Legitimate (1)'],
                output_dict=True,
                zero_division=0
            )

            formatted_cls_report = []
            for label in ['Phishing (0)', 'Legitimate (1)']:
                if label in cls_report_raw:
                    formatted_cls_report.append({
                        'class_name': label,
                        'precision': cls_report_raw[label]['precision'],
                        'recall': cls_report_raw[label]['recall'],
                        'f1_score': cls_report_raw[label]['f1-score'],
                        'support': cls_report_raw[label]['support']
                    })

            # Sample ROC points (max 25 points for smooth frontend chart)
            sample_step = max(1, len(fpr) // 25)
            roc_points = [
                {"x": round(float(fpr[i]), 3), "y": round(float(tpr[i]), 3)}
                for i in range(0, len(fpr), sample_step)
            ]
            if not roc_points or roc_points[-1] != {"x": 1.0, "y": 1.0}:
                roc_points.append({"x": 1.0, "y": 1.0})

            report_data = {
                'total_samples': len(df),
                'accuracy': round(acc * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'roc_auc': round(roc_auc, 4),
                'cm': {
                    'tn': int(cm[0][0]),
                    'fp': int(cm[0][1]),
                    'fn': int(cm[1][0]),
                    'tp': int(cm[1][1])
                },
                'roc_points': roc_points,
                'cls_report': formatted_cls_report,
                'file_name': csv_file.name
            }

        except Exception as e:
            error_message = f"Training & Validation error: {str(e)}"

    return render(request, 'admin/modeltraining.html', {
        'report': report_data,
        'error_message': error_message
    })

def exsistingnotebook(request):
    return render(request, 'admin/exsistingnotebook.html')

def proposednotebook(request):
    return render(request, 'admin/proposednotebook.html')