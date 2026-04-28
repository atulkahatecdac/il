import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# --- MACHINE LEARNING: SYSTEM LEARNS FROM DATA ---

# 1. Historical Data (Study Hours, Attendance %)
X_train = np.array([
    [2, 40],  # Student A
    [3, 50],  # Student B
    [5, 70],  # Student C
    [6, 80],  # Student D
    [8, 90]   # Student E
])

# 2. Historical Results (0 = Fail, 1 = Pass)
y_train = np.array([0, 0, 1, 1, 1])

# 3. Train the Model (The machine learns the pattern)
ml_model = KNeighborsClassifier(n_neighbors=1)
ml_model.fit(X_train, y_train)

def ml_system(study_hours, attendance_percentage):
    # The model predicts based on past patterns
    prediction = ml_model.predict([[study_hours, attendance_percentage]])
    return "Pass" if prediction[0] == 1 else "Fail"

# --- LIVE TEST: The Edge Case ---
# The same student: 8 hours, 30% attendance.
test_hours = 8
test_attendance = 30

print("--- Machine Learning Output ---")
print(f"Testing Student: {test_hours} Hours, {test_attendance}% Attendance")
print(f"System Prediction: {ml_system(test_hours, test_attendance)}")
print("\nWhy it succeeded: The ML model compared this student to past data. It recognized the 30% attendance pattern closely matches historical failures.")

import matplotlib.pyplot as plt

# --- VISUALIZING THE MACHINE LEARNING LOGIC ---

plt.figure(figsize=(9, 6))

# 1. Plot Historical Fails (Red dots)
plt.scatter([2, 3], [40, 50], color='red', s=120, label='Historical Fails')

# 2. Plot Historical Passes (Green dots)
plt.scatter([5, 6, 8], [70, 80, 90], color='green', s=120, label='Historical Passes')

# 3. Plot the Live Edge Case (Blue Star)
plt.scatter([8], [30], color='blue', marker='*', s=350, label='Edge Case (8h, 30%)')

# 4. Draw the "Traditional Rule" boundary
plt.axvline(x=6, color='gray', linestyle='--', label='Traditional Rule: 6+ Hours')

# Formatting the graph
plt.title('Why the ML Model Chose "Fail"', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Attendance %', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the graph
plt.show()