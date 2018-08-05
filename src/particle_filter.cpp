/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (is_initialized == true)
  {
    return;
  }

  num_particles = 50; /* Seems to be enough */

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0f;
    particles.push_back(p);
    p.id = i;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  // define normal distributions for sensor noise

  normal_distribution<double> x_noise(0, std_pos[0]);
  normal_distribution<double> y_noise(0, std_pos[1]);
  normal_distribution<double> theta_noise(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {
    double ang_speed, yaw_angle, displacement;
    /* Predict using the bicycle motion model */
    if (fabs(yaw_rate) > 0.0001)
    {
      ang_speed = velocity / yaw_rate;
      yaw_angle = yaw_rate * delta_t;
      particles[i].x += (ang_speed * (sin(particles[i].theta + yaw_angle) - sin(particles[i].theta)));
      particles[i].y += (ang_speed * (cos(particles[i].theta) - cos(particles[i].theta + yaw_angle)));
      particles[i].theta += yaw_angle;
    }
    else
    {
      displacement = velocity * delta_t;
      /* No change in heading in this case */
      particles[i].x += (displacement * cos(particles[i].theta));
      particles[i].y += (displacement * sin(particles[i].theta));
    }
    particles[i].x += x_noise(gen);
    particles[i].y += y_noise(gen);
    particles[i].theta += theta_noise(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (unsigned int i = 0; i < observations.size(); i++)
  {
    LandmarkObs o = observations[i];

    double min_dist = numeric_limits<double>::max();

    int map_id = -1;

    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      LandmarkObs p = predicted[j];

      /* Get distance between predicted and observed landmark */
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      /* If closer than remember */
      if (cur_dist < min_dist)
      {
        min_dist = cur_dist;
        map_id = p.id;
      }
    }
    /* store best id */
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {

    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    /* Loop through all the landmarks, and only consider the ones that are within the sensor range */
    vector<LandmarkObs> lm_inrange;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range)
      {
        lm_inrange.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    /* Transform all observations, that are coming in sensor coordinates, into map coordinates */
    vector<LandmarkObs> map_obs;
    for (unsigned int j = 0; j < observations.size(); j++)
    {
      double t_x = (cos(p_theta)*observations[j].x) - (sin(p_theta)*observations[j].y) + p_x;
      double t_y = (sin(p_theta)*observations[j].x) + (cos(p_theta)*observations[j].y) + p_y;
      map_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    /* Perform the data association step */
    dataAssociation(lm_inrange, map_obs);

    /* Recompute weights */
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < map_obs.size(); j++)
    {

      double obs_x, obs_y, pred_x, pred_y;
      obs_x = map_obs[j].x;
      obs_y = map_obs[j].y;

      /* Get the x,y coordinates of the matched landmark */
      for (unsigned int k = 0; k < lm_inrange.size(); k++)
      {
        if (lm_inrange[k].id == map_obs[j].id)
        {
          pred_x = lm_inrange[k].x;
          pred_y = lm_inrange[k].y;
        }
      }

      /* MVG */
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double norm_fct = (1 / (2 * M_PI*(s_x*s_y)));
      double temp1 = pow(pred_x - obs_x, 2) / (2 * pow(s_x, 2));
      double temp2 = pow(pred_y - obs_y, 2) / (2 * pow(s_y, 2));
      double obs_w = (norm_fct * exp(-(temp1 + temp2)));
      /* Update particle weight */
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> resampled_particles;

  
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }

  /* random index for the "wheel" */ 
  uniform_int_distribution<int> uniintdist(0, num_particles - 1);
  auto index = uniintdist(gen);

  
  double max_weight = *max_element(weights.begin(), weights.end());

  
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  /* Apply "wheel" technique from lesson */
  for (int i = 0; i < num_particles; i++) 
  {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) 
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
