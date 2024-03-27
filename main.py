#
# import pygame
# from OpenGL.GL import *
# from OpenGL.GLU import *
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Define the SDF function
#
#         # Create mesh grid
#         # grid_size = 50
#         # x_range = np.linspace(-200, 200, grid_size)
#         # y_range = np.linspace(-200, 200, grid_size)
#         # z_range = np.linspace(-200, 200, grid_size)
#         # x_mesh, y_mesh, z_mesh = np.meshgrid(x_range, y_range, z_range)
#
#         # Pygame initialization
#         pygame.init()
#         width, height = 800, 600
#         pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
#         glViewport(0, 0, width, height)
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         gluPerspective(45, (width / height), 0.1, 50.0)
#         glMatrixMode(GL_MODELVIEW)
#         glLoadIdentity()
#         gluLookAt(0, 0, -5, 0, 0, 0, 0, 1, 0)
#         clock = pygame.time.Clock()
#
#         # Main loop
#         running = True
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#
#             glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#             glPushMatrix()
#
#             # Render SDF points as a point cloud
#             glBegin(GL_POINTS)
#             for index in range(len(obstacle_points)):
#                 pnt = obstacle_points[index]
#                 sdf_value = sdf_function(pnt[0],pnt[1], pnt[2])
#                 if sdf_value[0] <= 0:  # Render only points inside the surface
#                     glColor3f(1, 1, 1)
#                     glVertex3f(pnt[0], pnt[1], pnt[2])
#                     print(pnt[0], pnt[1], pnt[2])
#
#             # for i in range(grid_size):
#             #     for j in range(grid_size):
#             #         for k in range(grid_size):
#             #             x, y, z = x_mesh[i, j, k], y_mesh[i, j, k], z_mesh[i, j, k]
#             #             sdf_value = sdf_function(x, y, z)
#             #             if sdf_value[0] <= 0:  # Render only points inside the surface
#             #                 glColor3f(1, 1, 1)
#             #                 glVertex3f(x, y, z)
#             glEnd()
#
#             glPopMatrix()
#             pygame.display.flip()
#             clock.tick(30)
#
#         pygame.quit()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# # # Save point cloud as .obj file
# # obj_filename = '/Users/ak/PycharmProjects/SDF/LIDARDATA/point_cloud.obj'
# # with open(obj_filename, 'w') as obj_file:
# #     for point in points_coordinates:
# #         obj_file.write(f'v {point[0]} {point[1]} {point[2]}\n')
# #
# # # Save point cloud as .txt file
# # txt_filename = '/Users/ak/PycharmProjects/SDF/LIDARDATA/point_cloud.txt'
# # np.savetxt(txt_filename, points_coordinates, fmt='%.6f')
#
#
# # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
# # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', render_intensity=True)
# # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT')
# my_sample = nusc.sample[19]
# #nusc.render_sample_data(my_sample['data']['CAM_FRONT'])
# nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=8, underlay_map=True)
# nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=60, underlay_map=True)
# #nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)
# from nuscenes.utils.data_classes import RadarPointCloud
# RadarPointCloud.disable_filters()
# #nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)
# RadarPointCloud.default_filters()
# nusc.render_annotation(my_sample['anns'][22])
# my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
# nusc.render_scene(my_scene_token)
# nusc.render_egoposes_on_map(log_location='singapore-onenorth')
