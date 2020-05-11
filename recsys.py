def main():
    movieData = loadCSV("movieData/movies_metadata.csv")
    #simpleRec(movieData)
    plotBasedRec(movieData, "The Godfather")


import numpy as np


def simpleRec(movieData):
    def rating(movie, meanV, minVC):
        vc = int(movie['vote_count'])
        av = float(movie['vote_average'])
        return (vc / (vc + minVC) * av) + (meanV / (meanV + vc) * meanV)

    meanVote = sum(float(x['vote_average'])
                   for x in movieData) / len(movieData)

    minVotes = np.percentile([int(x['vote_count']) for x in movieData], 90)

    sortedBest = sorted(
        [x for x in movieData if int(x['vote_count']) > minVotes],
        key=lambda x: rating(x, meanVote, minVotes),
        reverse=True)

    for movie in sortedBest[0:10]:
        print(movie['title'])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def plotBasedRec(movieData, movieTitle):
    tfidf = TfidfVectorizer(stop_words='english')

    tfidfMatrix = tfidf.fit_transform(x['overview'] for x in movieData)
    cosineMatrix = linear_kernel(tfidfMatrix, tfidfMatrix)

    titleIndex = [
        idx for idx, movie in enumerate(movieData)
        if movie['title'] == movieTitle
    ][0]

    sortedScores = sorted(enumerate(cosineMatrix[titleIndex]),
                          key=lambda x: x[1],
                          reverse=True)

    print([movieData[x[0]]['title'] for x in sortedScores[0:10]])


import csv


def loadCSV(path):
    data = []
    with open(path) as f:
        cols = next(f).strip().split(",")
        for row in csv.reader(f, delimiter=","):
            if len(row) == 24:
                data.append(dict(zip(cols, row)))
    return data


if __name__ == "__main__":
    main()
