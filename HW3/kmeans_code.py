import concurrent.futures
import numpy as np
import pandas as pd
import time

def lists_equal(value_1, value_2):
    return np.array_equal(np.sort(value_1), np.sort(value_2))

#using Euclidean distance for distance calculation
def euclidean(data_point1, data_point2):
    try:
        distance = 0
        for a,b in zip(data_point1, data_point2):
            euclidean_distance += pow((a-b), 2)
        return math.sqrt(euclidean_distance)
    except Exception as e:
        print(e)
        print(x)
        print(y)


#using jaccard distance for distance calculation
def jaccard(value_1, value_2):
    try:
        return 1 - np.divide(np.sum(np.minimum(value_1, value_2)),
                         np.sum(np.maximum(value_1, value_2)))
    except Exception as e:
        print(e)
        print(x)
        print(y)

#using cosine distance for distance calculation
def cosine(value_1, value_2):
    try:
        return 1 - np.divide(np.sum(np.multiply(value_1, value_2)),
                         np.multiply(np.sqrt(np.sum(np.square(value_1))),
                                     np.sqrt(np.sum(np.square(value_2)))))
    except Exception as e:
        print(e)
        print(x)
        print(y)

def SSE(dist_func, X_data_frame, centroids):
    SSE_result = 0
    for centroid in centroids:
        for point in X_data_frame:
            SSE_result += dist_func(centroid, point)**2
    return SSE_result


def kMeans(distance_func, X_data_frame, Y_data_frame=[], K=0, centroids=np.array([]), stoppers=["Centroid stays the same"],
           maxIterations=0, task_id=""):

    if task_id == "task_1_stop":
        retrieve = str(distance_func.__name__) + "\t" + str(stoppers)
    else:
        retrieve = str(distance_func.__name__) + "\t"
   
    Y_data_frame_computed = np.full(X_data_frame.shape[0], 0)
   
    if len(stoppers) < 1 and maxIterations == 0:
        print("Stop criteria missing")
        return
  
    if centroids.size > 0:

        if len(centroids) != K:
            print("Mismatch: Found " + str(centroids) +
                  "Initial centroids with K = " + str(K))
 
    else:
        centroids = X_data_frame[np.random.choice(X_data_frame.shape[0], K, replace=False), :]

    start = time.time_ns()
    iterations = 0
    while True:

        old_centroids = np.copy(centroids)
        iterations += 1

        tmp_centroid_sum = np.zeros(centroids.shape)
        temp_centroid_count = np.zeros(centroids.shape[0])
    
        for point_idx, point in enumerate(X_data_frame):
            shortest_distance = float('inf')
  
            for centroid_idx, centroid in enumerate(centroids):
                distance = distance_func(point, centroid)
                if distance < shortest_distance:
                    shortest_distance = distance

                    Y_data_frame_computed[point_idx] = centroid_idx

            tmp_centroid_sum[Y_data_frame_computed[point_idx]] = np.add(
                tmp_centroid_sum[Y_data_frame_computed[point_idx]], point)

            temp_centroid_count[Y_data_frame_computed[point_idx]] += 1

        for i in range(len(centroids)):

            if temp_centroid_count[i] == 0:
                print("Centroid discovered empty during iterations " + str(iterations))

                centroids[i] = np.copy(old_centroids[i])
            else:
                centroids[i] = np.divide(tmp_centroid_sum[i],
                                         np.full(centroids.shape[1], temp_centroid_count[i]))

        if "Centroid stays the same" in stoppers and lists_equal(old_centroids, centroids):
            break #when the centroid position stays unchanged
           
        if "SSE" in stoppers and SSE(distance_func, X_data_frame, centroids) \
                > SSE(distance_func, X_data_frame, old_centroids):

            centroids = np.copy(old_centroids)
            break #if the SSE value rises in the following iteration

        if (maxIterations != 0 and iterations >= maxIterations) \
                or (maxIterations == 0 and iterations >= 500):
            break #when the maximum iteration values reaches

    end = time.time_ns()

    if task_id == "task_1":
        retrieve += "SSE = " + str(SSE(distance_func, X_data_frame, centroids)) + "\n"
        retrieve += "Predictive accuracy = " + str(accuracy(Y_data_frame, Y_data_frame_computed))
    if task_id == "task_1_stop":
        retrieve += "\t" + str(iterations) + "\t" + str(SSE(distance_func, X_data_frame, centroids)) \
            + "\t" + str(end - start) + " nano_seconds"
    return retrieve

def accuracy(Y_data_frame, Y_data_frame_computed):
    
    # Labeling each cluster based on the data points' labels that received the most votes.
    cluster_score = []
    for i in range(len(Y_data_frame)):
        cluster_score.insert(i, [])
        for j in range(len(Y_data_frame)):
            cluster_score[i].insert(j, 0)
    
    for i in range(len(Y_data_frame)):
        cluster_score[Y_data_frame_computed[i]][Y_data_frame[i][0]] += 1
    
    correct_cluster_score = 0
    total_cluster_score = 0
    for i in range(len(Y_data_frame)):
        winner = 0
        maximum_seen_value = 0
        for j in range(len(Y_data_frame)):
            if cluster_score[i][j] > maximum_seen_value:
                winner = j
                maximum_seen_value = cluster_score[i][j]
                
        for j in range(len(Y_data_frame)):
            total_cluster_score += cluster_score[i][j]
            if j == winner:
                correct_cluster_score += cluster_score[i][j]
    return ((correct_cluster_score / total_cluster_score) * 100)


def main():

    X_data_frame = pd.read_csv("data.csv")
    Y_data_frame = pd.read_csv("label.csv")
    X_data_frame = X_data_frame.to_numpy(dtype=float)
    Y_data_frame = Y_data_frame.to_numpy(dtype=int)

  
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        
        future_process = []
        
        # TASK 1 -> Q1 and Q2
        future_process.append(executor.submit(
            kMeans, euclidean, X_data_frame, Y_data_frame=Y_data_frame, K=10, task_id="task_1"))
        future_process.append(executor.submit(
            kMeans, cosine, X_data_frame, Y_data_frame=Y_data_frame, K=10, task_id="task_1"))
        future_process.append(executor.submit(
            kMeans, jaccard, X_data_frame, Y_data_frame=Y_data_frame, K=10, task_id="task_1"))

        # TASK 1 -> Q3
        future_process.append(executor.submit(kMeans, euclidean, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, maxIterations=100, stoppers=["Centroid stays the same", "SSE"], task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, cosine, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, maxIterations=100, stoppers=["Centroid stays the same", "SSE"], task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, jaccard, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, maxIterations=100, stoppers=["Centroid stays the same", "SSE"], task_id="task_1_stop"))

        # TASK 1 -> Q4
        # Running K_means for the no change in centroid
        future_process.append(executor.submit(kMeans, euclidean, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=["Centroid stays the same"], task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, cosine, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=["Centroid stays the same"], task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, jaccard, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=["Centroid stays the same"], task_id="task_1_stop"))

        # Running K_means for when the SSE value increases in the next iteration
        future_process.append(executor.submit(kMeans, euclidean, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=["SSE"], task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, cosine, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=["SSE"], task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, jaccard, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=["SSE"], task_id="task_1_stop"))

        # Running K_means till the maximum iteration occurs
        future_process.append(executor.submit(kMeans, euclidean, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=[], maxIterations=100, task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, cosine, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=[], maxIterations=100, task_id="task_1_stop"))
        future_process.append(executor.submit(kMeans, jaccard, X_data_frame, Y_data_frame=Y_data_frame,
                       K=10, stoppers=[], maxIterations=100, task_id="task_1_stop"))

        
        future_iterations = iter(future_process)

        print("\nTask Q1 and Task Q2 \n")
        print(next(future_iterations).result())
        print("\n")
        print(next(future_iterations).result())
        print("\n")
        print(next(future_iterations).result())

        print("\nTask Q3 \n")
        print("Distance\tStopping Criteria\t\t\tIterations\tSSE\tTime")
        print("\n")
        print(next(future_iterations).result())
        print(next(future_iterations).result())
        print(next(future_iterations).result())

        print("\nTask Q4 \n")
        print("Distance\tStopping Criteria\t\t\tIterations\tSSE\tTime")
        print("\n")
        print(next(future_iterations).result())
        print(next(future_iterations).result())
        print(next(future_iterations).result())
        print("\n")
        print(next(future_iterations).result())
        print(next(future_iterations).result())
        print(next(future_iterations).result())
        print("\n")
        print(next(future_iterations).result())
        print(next(future_iterations).result())
        print(next(future_iterations).result())

if __name__ == "__main__":
    main()