years = [2020]
months = [1]
days = [11]#, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


for year in years:
    
    for im, month in enumerate(months):
        
        for day in range(1, days[im-1] + 1):

            print(year, month, day)