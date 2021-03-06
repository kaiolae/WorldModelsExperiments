FIREBALL_THRESHOLD = 0.5
from count_monsters_and_balls import count_monsters, count_fireballs

#TODO May also need a method that takes two sequences (real and dreamed) and measures DIFFERENCES.
def count_events_from_images(image_sequence):
    num_fireballs = 0
    num_monsters = 0
    thresholded_images = [] #Potentially useful for debugging
    for img in image_sequence:
        fb, thresholded_image = count_fireballs(img, FIREBALL_THRESHOLD)
        thresholded_images.append(thresholded_image)

        num_fireballs+=fb

        monsters, img = count_monsters(img)
        num_monsters += monsters

    return {"num_fireballs":num_fireballs, "num_monsters":num_monsters}

def count_appearances_and_disappearances_from_images(image_sequence):
    fireball_delta = 0
    monster_delta = 0
    thresholded_images = [] #Potentially useful for debugging
    for img_counter in range(len(image_sequence)):
        if img_counter==0:
            continue
        fb_after, thresholded_image = count_fireballs(image_sequence[img_counter], FIREBALL_THRESHOLD)
        fb_before, thresholded_image = count_fireballs(image_sequence[img_counter-1], FIREBALL_THRESHOLD)
        thresholded_images.append(thresholded_image)

        fireball_delta+=abs(fb_after-fb_before)

        monsters_after, thresholded_image = count_monsters(image_sequence[img_counter])
        monsters_before, thresholded_image = count_monsters(image_sequence[img_counter-1])
        monster_delta += abs(monsters_after-monsters_before)

    return {"fireball_delta":fireball_delta, "monster_delta":monster_delta}


def count_different_events_in_images(real_images, predicted_images):
    #TODO: Note it's important the caller aligns these, so the prediction for t=0 and real event at t=0 are both at index 0 in arrays.
    assert(len(real_images) == len(predicted_images))
    missing_fireballs = 0
    imagined_fireballs = 0
    missing_monsters = 0
    imagined_monsters = 0
    for i in range(len(real_images)):
        actual_num_fireballs, img = count_fireballs(real_images[i], FIREBALL_THRESHOLD)
        predicted_num_fireballs, thresholded_image = count_fireballs(predicted_images[i],FIREBALL_THRESHOLD)

        if actual_num_fireballs>predicted_num_fireballs:
            missing_fireballs+=actual_num_fireballs-predicted_num_fireballs
        elif predicted_num_fireballs>actual_num_fireballs:
            imagined_fireballs+=predicted_num_fireballs-actual_num_fireballs

        actual_num_monsters, img = count_monsters(real_images[i])
        predicted_num_monsters, img = count_monsters(predicted_images[i])
        if actual_num_monsters>predicted_num_monsters:
            missing_monsters+=actual_num_monsters-predicted_num_monsters
        elif predicted_num_monsters>actual_num_monsters:
            imagined_monsters+=predicted_num_monsters-actual_num_monsters

    return {"missing_fireballs": missing_fireballs, "imagined_fireballs": imagined_fireballs, "missing_monsters":missing_monsters,
            "imagined_monsters": imagined_monsters}

def count_events_on_trained_rnn(trained_vae, trained_rnn, initial_latent_vector, actions, num_timesteps = 100):
    assert(len(actions)>=num_timesteps)
    dreamed_latents = []
    dreamed_latent = trained_rnn.predict_one_step(actions[0], previous_z=initial_latent_vector)
    dreamed_latents.append(dreamed_latent)
    for i in range(num_timesteps-1):
        dreamed_latents.append(trained_rnn.predict_one_step(actions[i+1]))

    predicted_images = trained_vae.decode(dreamed_latents)

    return count_events_from_images(predicted_images)

def count_appearances_and_disappearances(trained_vae, trained_rnn, initial_latent_vector, actions, num_timesteps = 100):
    assert(len(actions)>=num_timesteps)
    dreamed_latents = []
    dreamed_latent = trained_rnn.predict_one_step(actions[0], previous_z=initial_latent_vector)
    dreamed_latents.append(dreamed_latent)
    for i in range(num_timesteps-1):
        dreamed_latents.append(trained_rnn.predict_one_step(actions[i+1]))

    predicted_images = trained_vae.decode(dreamed_latents)

    return count_appearances_and_disappearances_from_images(predicted_images)

def count_differences_between_reality_and_prediction(trained_vae, trained_rnn, real_latent_sequence, actions):
    #real latent sequences: the N observations. Actions: The N-1 actions BETWEEN those observations.
    assert(len(actions)>= len(real_latent_sequence)-1)
    real_images = trained_vae.decode(real_latent_sequence)
    dreamed_latents = []
    for i in range(len(real_latent_sequence)-1):
        dreamed_latents.append(trained_rnn.predict_one_step(actions[i], previous_z=real_latent_sequence[i]))

    dreamed_images = trained_vae.decode(dreamed_latents) #The predictions for the NEXT image after the N-1 first observations.

    #Lining up the predictions with the actual timestep they predict here.
    return count_different_events_in_images(real_images[1:], dreamed_images)
