def lagrange_interpolation_function(data_points, x):
    y = 0
    for idx, data_point in enumerate(data_points):
        intermidiate_y = data_point[1]
        
        for idx2, data_point2 in enumerate(data_points):
            if idx == idx2:
                continue
            intermidiate_y *= (x - data_point2[0]) / (data_point[0] - data_point2[0])

        y += intermidiate_y

    return y
