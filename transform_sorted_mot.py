def transform_line(line):
    parts = line.strip().split(',')
    frame_id = int(parts[0])
    x = parts[1]
    y = parts[2]
    obj_id = int(parts[3])
    
    # 创建新的格式 [0, frame_id, id, -1, -1, -1, -1, 1, x, y, -1]
    new_line = f"0,{frame_id},{obj_id},-1,-1,-1,-1,1,{x},{y},-1"
    return frame_id, obj_id, new_line

def transform_and_sort_file(input_file, output_file):
    lines_dict = {}
    
    with open(input_file, 'r') as infile:
        for line in infile:
            frame_id, obj_id, new_line = transform_line(line)
            if frame_id not in lines_dict:
                lines_dict[frame_id] = []
            lines_dict[frame_id].append((obj_id, new_line))
    
    with open(output_file, 'w') as outfile:
        for frame_id in sorted(lines_dict.keys()):
            # 按照 id 从小到大排序
            sorted_lines = sorted(lines_dict[frame_id], key=lambda x: x[0])
            for _, new_line in sorted_lines:
                outfile.write(new_line + '\n')

# 输入文件和输出文件路径
input_file = '/home/SENSETIME/lizirui/utils/pts_tracker/pred_40_track_result.txt'
output_file = '/home/SENSETIME/lizirui/utils/pts_tracker/pred_40_track_result_mot.txt'

# 转换并排序文件
transform_and_sort_file(input_file, output_file)
