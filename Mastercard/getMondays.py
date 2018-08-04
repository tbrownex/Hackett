# Build a dictionary mapping a month to the number of Mondays in the month
import pandas as pd
import calendar

def getCount(month):
    count = 0
    day, days = calendar.monthrange(month.year,month.month)
    if day < 2:       # Monday, Tuesday
        count = 4
    else:
        if day == 2:     # Wednesday
            if days == 31:
                count = 5
            else:
                count = 4
        else:
            if day == 3:     # Thursday
                if days > 29:
                    count = 5
                else:
                    count = 4
            else:     # Friday, Saturday, Sunday
                if month.month == 2:
                    count = 4
                else:
                    count = 5
    return count

def monthToMondays():
    dates  = pd.date_range("2013-01-01", periods=66, freq="m")
    array  = dates.values.astype('datetime64[M]')
    months = pd.DataFrame(array)
    
    # Now get the number of Mondays associated with each month
    monthMap = {}
    for m in months[0]:
        mondays = getCount(m)
        m = str(m.date())
        monthMap[m] = mondays
    return monthMap