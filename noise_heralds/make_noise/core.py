import matplotlib.pyplot as plt
from models.scenario import Scenario
from noise_heralds.make_noise.clustering import calculate_front_points, cluster_front_points, \
    calculate_clusters_centers, \
    create_centers_graph, calculate_min_set_cover
from noise_heralds.make_noise.post_process import push_centers_out_of_contagion_polygon, \
    move_clusters_centers_towards_patient
from noise_heralds.segment import SegmentedAreas


def place_heralds_main(seg_area: SegmentedAreas):
    plt.figure(figsize=(12, 10))

    # assign the boundary and patient location to variables
    villages_boundary = seg_area.villages.boundary
    scenario = Scenario(heralds=None)

    # calculate all front points (points directed at the patient)
    front_points = calculate_front_points(scenario.patient.location, villages_boundary)

    # cluster the front points
    clusters_labels_dict = cluster_front_points(front_points)

    # calculate cluster centers from clusters
    clusters_centers = calculate_clusters_centers(clusters_labels_dict)

    # move the centers towards the patient location
    move_clusters_centers_towards_patient(clusters_centers, scenario.patient.location)

    # move the centers towards the patient location
    push_centers_out_of_contagion_polygon(clusters_centers, scenario)

    # creates a clusters centers graph - an edge exists if the intersection is nonzero
    G = create_centers_graph(clusters_centers)

    # calculates min set cover on the clusters centers graph
    min_set_labels = calculate_min_set_cover(clusters_centers, G)

    # filter to the min set clusters
    filtered_clusters_centers = {label: clusters_centers[label] for label in min_set_labels}

    noise_output = {"filtered_clusters_centers": filtered_clusters_centers}

    return noise_output
