figure_mosaic = """
ABFI
ACGI
ADHI
AEHI
"""

fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(11, 5))

axes["A"].plot()
