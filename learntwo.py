import pandas as pd
import numpy as np

employees = pd.DataFrame({
    'employee_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'department_id': [1, 2, 1, 3]
})

departments = pd.DataFrame({
    'department_id': [1, 2, 3],
    'department_name': ['HR', 'Engineering', 'Sales']
})

print("--- Employees ---")
print(employees)
print("\n--- Departments ---")
print(departments)

merge_df = pd.merge(employees,departments,on='department_id')
print("Merged data")
print(merge_df)


ts = pd.Timestamp('2025-10-22 11:30:00')
print(f"Single Timestamp: {ts}")
print(f"Timestamp Day: {ts.day_name()}")
date_str = "2025-10-23"
dt_obj = pd.to_datetime(date_str)
print(f"\nConverted Datetime: {dt_obj}")

date_index = pd.date_range(start='2025-10-22', end='2025-10-28', freq='D')
print(f"\nDate Range: {date_index}")

data = {
    'Date': ['2025-10-22', '2025-10-23', '2025-10-24', '2025-10-25'],
    'Value': [10, 15, 12, 18]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(f"DataFrame with datetime index:\n {df}")

#create sample hourly data
dates = pd.date_range(start='2025-10-22', periods=48, freq='h')
values = np.random.rand(48) * 100
df_hourly = pd.DataFrame({'Value': values}, index=dates)
print("Hourly Data")
print(df_hourly.head())

df_daily = df_hourly.resample('D').mean()
print(df_daily)