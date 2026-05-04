from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
import sys
import os

from .models import CareerResult

# -----------------------------
# ADD ML PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PATH = os.path.join(BASE_DIR, '../ml')
sys.path.append(ML_PATH)

from predict import predict_career


# -----------------------------
# HOME PAGE
# -----------------------------
def home(request):
    return render(request, 'form.html')


# -----------------------------
# PREDICTION VIEW
# -----------------------------
def predict_view(request):
    if request.method == 'POST':

        # ✅ GET INPUTS SAFELY
        skills = request.POST.get('skills', '').strip()
        education = request.POST.get('education', '').strip()

        # ✅ EXPERIENCE VALIDATION
        try:
            experience = int(request.POST.get('experience', 0))
        except ValueError:
            return render(request, 'form.html', {
                'error': 'Please enter valid experience (number only)'
            })

        # ✅ EDUCATION VALIDATION (MATCH DROPDOWN)
        valid_education = ["B.Sc", "B.Tech", "B.Com", "MBA", "MCA"]

        if education not in valid_education:
            return render(request, 'form.html', {
                'error': 'Please select valid education'
            })

        # ✅ EMPTY FIELD CHECK
        if not skills:
            return render(request, 'form.html', {
                'error': 'Please enter your skills'
            })

        # -----------------------------
        # CALL ML MODEL
        # -----------------------------
        try:
            result = predict_career(skills, experience, education)
        except Exception as e:
            return render(request, 'form.html', {
                'error': f"Prediction error: {str(e)}"
            })

        # -----------------------------
        # SAVE RESULT (IF LOGGED IN)
        # -----------------------------
        if request.user.is_authenticated:
            CareerResult.objects.create(
                user=request.user,
                role=result.get("Role"),
                salary=result.get("Salary_LPA"),
                cluster=result.get("Cluster")
            )

        # -----------------------------
        # SHOW RESULT
        # -----------------------------
        return render(request, 'result.html', {
            'result': result,
            'user_skills': skills,
            'experience': experience,
            'education': education
        })

    return render(request, 'form.html')


# -----------------------------
# LOGIN VIEW
# -----------------------------
def user_login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'login.html', {
                'error': 'Invalid username or password'
            })

    return render(request, 'login.html')


# -----------------------------
# HISTORY VIEW
# -----------------------------
def history(request):
    if request.user.is_authenticated:
        data = CareerResult.objects.filter(user=request.user)
    else:
        data = []

    return render(request, "history.html", {"data": data})