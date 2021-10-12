import numpy as np
# For city block distance
from scipy.spatial import distance

# Input data
sample = np.loadtxt("1007test.txt", delimiter=',')
trainer = np.loadtxt("optdigits.tra", delimiter=',')

# Create 4 3-d arrays which store differnt distance arithmetics.
# x-axis for digital, y-axis for samples, z-axis for each instance's class
eudarr = np.zeros(shape=(10, 3823, 2), dtype=float)
manarr = np.zeros(shape=(10, 3823, 2), dtype=float)
chebyarr = np.zeros(shape=(10, 3823, 2), dtype=float)
cosarr = np.zeros(shape=(10, 3823, 2), dtype=float)


class distance_calculator:
    def __init__(self, dimA, dimB):
        self.dima = dimA
        self.dimb = dimB
        # The 65th dimension is not to be included but a class.
        self.dima_winthoutclass = np.delete(self.dima, -1, axis=1)
        self.dimb_winthoutclass = np.delete(self.dimb, -1, axis=1)

    def return_distance_data(self):
        d_count = 0
        # Each digital needs to be computed.
        for i in self.dima_winthoutclass:
            j_count = 0
            # How many instances the trainer has.
            for j in self.dimb_winthoutclass:
                # For correlation(cos)
                cosarr[d_count][j_count][0] = distance.correlation(i, j)
                cosarr[d_count][j_count][1] = self.dimb[j_count][64]
                # For euclidean
                eudarr[d_count][j_count][0] = np.linalg.norm(j-i)
                eudarr[d_count][j_count][1] = self.dimb[j_count][64]
                # For manhattan
                manarr[d_count][j_count][0] = distance.cityblock(i, j)
                manarr[d_count][j_count][1] = self.dimb[j_count][64]
                # For chebyshev
                chebyarr[d_count][j_count][0] = distance.chebyshev(i, j)
                chebyarr[d_count][j_count][1] = self.dimb[j_count][64]
                j_count += 1
            d_count += 1


dc1 = distance_calculator(sample, trainer)
dc1.return_distance_data()

# sort by column[0]
for i in range(0, len(eudarr)):
    eudarr[i] = eudarr[i][eudarr[i][:, 0].argsort()]
    manarr[i] = manarr[i][manarr[i][:, 0].argsort()]
    chebyarr[i] = chebyarr[i][chebyarr[i][:, 0].argsort()]
    cosarr[i] = cosarr[i][cosarr[i][:, 0].argsort()]

pass
