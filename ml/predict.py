import pickle
import numpy as np

# -----------------------------
# 1. LOAD MODELS
# -----------------------------
with open('../models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('../models/regressor.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('../models/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('../models/edu_encoder.pkl', 'rb') as f:
    edu_encoder = pickle.load(f)

with open('../models/role_encoder.pkl', 'rb') as f:
    role_encoder = pickle.load(f)

# -----------------------------
# 2. SKILL GAP DICTIONARY
# -----------------------------
required_skills = {
    "Data Analyst": ["python", "sql", "excel", "powerbi"],
    "Backend Developer": ["python", "django", "api", "sql"],
    "ML Engineer": ["python", "ml", "tensorflow", "statistics"],
    "Frontend Developer": ["html", "css", "javascript", "react"]
}

# -----------------------------
# 3. EDUCATION CLEANING FUNCTION
# -----------------------------
def normalize_education(education):

    education = education.strip().lower()

    mapping = {
        "bsc": "B.Sc",
        "b.sc": "B.Sc",
        "bachelor of science": "B.Sc",

        "btech": "B.Tech",
        "b.tech": "B.Tech",
        "bachelor of technology": "B.Tech",

        "mba": "MBA",
        "mca": "MCA",

        "bcom": "B.Com",
        "b.com": "B.Com"
    }

    # convert to standard format
    education = mapping.get(education, education)

    # 🚨 HANDLE UNKNOWN VALUES (IMPORTANT)
    if education not in edu_encoder.classes_:
        # fallback to first known class (safe default)
        education = edu_encoder.classes_[0]

    return education


# -----------------------------
# 4. PREDICTION FUNCTION
# -----------------------------
def predict_career(skills, experience, education):

    # ✅ Clean inputs
    skills = skills.lower()
    education = normalize_education(education)

    # ✅ Transform
    skills_vec = vectorizer.transform([skills]).toarray()
    edu_encoded = edu_encoder.transform([education])

    X = np.hstack((skills_vec, [[experience, edu_encoded[0]]]))

    probs = clf.predict_proba(X)[0]

    # Top 3 roles
    top3_idx = probs.argsort()[-3:][::-1]

    top3_roles = []
    for i in top3_idx:
        role = role_encoder.inverse_transform([i])[0]
        confidence = round(probs[i] * 100, 2)
        top3_roles.append({"role": role, "confidence": confidence})

    job_role = top3_roles[0]["role"]

    salary = reg.predict(X)[0]
    cluster = kmeans.predict(X)[0]

    # -----------------------------
    # Skill Gap
    # -----------------------------
    user_skills = set(skills.split())
    needed_skills = set(required_skills.get(job_role, []))
    missing_skills = list(needed_skills - user_skills)

    cluster_meaning = {
        0: "Beginner Level",
        1: "Intermediate Level",
        2: "Advanced Level"
    }

    # Courses
    courses = {
        "python": "Learn Python - Coursera",
        "sql": "SQL Bootcamp - Udemy",
        "excel": "Excel for Data Analysis - Udemy",
        "django": "Django Full Course - YouTube",
        "ml": "Machine Learning - Andrew Ng",
        "tensorflow": "Deep Learning - Coursera",
        "react": "React JS - Udemy",
        "powerbi": "Power BI - Udemy"
    }

    recommended_courses = [courses.get(skill, skill) for skill in missing_skills]

    confidence = max(probs)

    roadmap = {
        "Beginner Level": "Start with basics → Learn core skills → Build small projects",
        "Intermediate Level": "Build real projects → Learn advanced tools → Improve problem-solving",
        "Advanced Level": "Apply for jobs → Contribute to real-world projects → Master specialization"
    }

    # Comparison
    comparison = []

    for role, skills_list in required_skills.items():
        match = len(set(skills_list) & user_skills)
        total = len(skills_list)
        score = round((match / total) * 100, 2)

        comparison.append({
            "role": role,
            "match": score
        })

    return {
        "Role": job_role,
        "Salary_LPA": round(salary, 2),
        "Cluster": cluster_meaning.get(int(cluster), "Unknown"),
        "missing_skills": missing_skills,
        "courses": recommended_courses,
        "confidence": round(confidence * 100, 2),
        "top3_roles": top3_roles,
        "roadmap": roadmap.get(cluster_meaning.get(int(cluster)), ""),
        "comparison": comparison,
    }


# -----------------------------
# 5. TEST RUN
# -----------------------------
if __name__ == "__main__":
    skills = input("Enter skills: ")
    experience = int(input("Enter experience (years): "))
    education = input("Enter education (e.g. BTech, BSc): ")

    result = predict_career(skills, experience, education)

    print("\n🎯 RESULT:")
    for key, value in result.items():
        print(f"{key}: {value}")