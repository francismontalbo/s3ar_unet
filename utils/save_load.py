#Save History Function
def save_h(save_folder, his):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_folder + save_name + '.history', 'wb') as file_pi:
        pickle.dump(his, file_pi)
    print("history saved")

def load_h(file):
    with open(save_folder + save_name + '.history', 'rb') as file_pi:
        his = pickle.load(file_pi)
    return his
