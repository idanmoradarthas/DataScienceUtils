from matplotlib import pyplot as plt, dates


@plt.FuncFormatter
def _convert_numbers_to_dates(x, pos):
    return dates.num2date(x).strftime("%Y-%m-%d %H:%M")
