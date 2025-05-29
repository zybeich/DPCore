import numpy as np

def create_cdc_sequence(distribution='dirichlet', delta=1.0, num_total_batches=5000//64 + 1):
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    num_domains = len(corruptions)
    domains  = list(range(num_domains))
    domain_order = []
    
    if distribution == 'multinomial':
        remaining_batches = {domain: num_total_batches for domain in domains}
        num_shift = 0
        while remaining_batches:
            selected_domains = np.random.choice(list(remaining_batches.keys()), 1)
            num_selected_batches = np.random.choice(list(range(1, remaining_batches[selected_domains[0]]+1)))
            remaining_batches[selected_domains[0]] -= num_selected_batches
            if remaining_batches[selected_domains[0]] == 0:
                del remaining_batches[selected_domains[0]]
            domain_order.extend([selected_domains[0]]*num_selected_batches)
            num_shift += 1
        
    elif distribution == 'dirichlet':
        slot_num = 3
        label_distribution = np.random.dirichlet([delta] * slot_num, 15)
        slot_indices = [[] for _ in range(slot_num)]
        class_indices = [ [i] * num_total_batches for i in range(15)]
        indices = np.arange(num_total_batches * 15)
        for c_ids, partition in zip(class_indices, label_distribution):
            for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                slot_indices[s].append(ids)

        for s_ids in slot_indices:
            permutation = np.random.permutation(range(len(s_ids)))
            ids = []
            for i in permutation:
                ids.extend(s_ids[i])
            domain_order.extend(indices[ids])
            
    return domain_order