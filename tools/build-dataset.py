from data_converter.nuscenes_converter_seg import  create_nuscenes_infos



if __name__ == '__main__':
    # Training settings
    create_nuscenes_infos( '/mnt/datasets/nuScenes/','HDmaps-final',version='v1.0-mini')

