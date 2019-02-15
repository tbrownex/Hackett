''' Input is a dataframe
    Apply a Min-Max scaler
    Not all columns should be normed, just the ones listed. There are some binary fields we'll skip
    '''
__author__ = "Tom Browne"

normCols = ["University Prestige", "Industry Yrs", "Role Yrs", "Current Job Level",\
            "Yrs Since Last Promotion", "Engagement_1", "Engagement_1", "Engagement_2",\
           "Engagement_3", "Engagement_4", "Engagement_5", "Autonomous_1", "Autonomous_1",\
           "Autonomous_2", "Autonomous_3", "Autonomous_4", "Autonomous_5", "Team Player_1",\
           "Team Player_2", "Team Player_3", "Team Player_4", "Team Player_5"]

def scaler(df):
    for col in normCols:
        df[col] = (df[col] - df[col].min()) / df[col].max()
    return df