# --- TRADITIONAL PROGRAMMING: HUMAN WRITES THE RULES ---

def rule_based_system(study_hours, attendance_percentage):
    # The programmer looks at the data and guesses a rule:
    # "Anyone studying 6 hours or more will pass."
    
    if study_hours >= 6:
        return "Pass"
    else:
        return "Fail"

# --- LIVE TEST: The Edge Case ---
# A student studies for 8 hours but only has 30% attendance.
test_hours = 8
test_attendance = 30

print("--- Traditional Programming Output ---")
print(f"Testing Student: {test_hours} Hours, {test_attendance}% Attendance")
print(f"System Prediction: {rule_based_system(test_hours, test_attendance)}")
print("\nWhy it failed: The human hardcoded a rule focusing only on hours, missing the nuance of low attendance.")