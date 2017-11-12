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
#include <math.h>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 150;
    for(int i = 0; i < num_particles; i++){
        weights.push_back(1.);
        Particle p;
        p.id = i;
        p.x = x;
        p.y = y;
        p.theta = theta;
        p.weight = 1.;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;
	std::normal_distribution<> dx(0.0, std_pos[0]);
	std::normal_distribution<> dy(0.0, std_pos[1]);
	std::normal_distribution<> dyaw(0.0, std_pos[2]);
    const double lin_vel = velocity * delta_t;
	const double curv_vel = velocity/yaw_rate;

	for(int i = 0; i < num_particles; i++){
        const double theta = particles[i].theta;
        const double next_theta = theta + yaw_rate*delta_t;

        if (std::abs(yaw_rate) < 0.000001){
            // straight motion

            particles[i].x += lin_vel * std::sin(theta);
            particles[i].y += lin_vel * std::cos(theta);
        }else{
            //turning motion

            particles[i].x += curv_vel * (std::sin(next_theta) - sin(theta));
            particles[i].y += curv_vel * (std::cos(theta) - cos(next_theta));
            particles[i].theta += yaw_rate*delta_t;
        }

        //add noise
        particles[i].x += dx(gen);
        particles[i].y += dy(gen);
        particles[i].theta += dyaw(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> near_lms, std::vector<LandmarkObs>& observations, double std_landmark[], int particle_idx) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    double dist_x = 0;
    double dist_y = 0;
    const double pre_coeff = 1./(2*M_PI*std_landmark[0]*std_landmark[1]);
    double x_exp = 0;
    double y_exp = 0;
    double exponent = 0;
    double curr_weight = 0;

    for(int j = 0; j < observations.size(); j++){
        double min_dist = std::numeric_limits<double>::max();
        double distance = 0;
        LandmarkObs nearest_lm;
        for(int i = 0; i < near_lms.size(); i++){
            //find min distance
            distance = dist(near_lms[i].x, near_lms[i].y, observations[j].x, observations[j].y);
            if (distance < min_dist){
                observations[j].id = near_lms[i].id;
                min_dist = distance;
                nearest_lm = near_lms[i];
            }
        }

        // also update the weights here so we don't need another iteration through the observations
        dist_x = observations[j].x - nearest_lm.x;
        dist_y = observations[j].y - nearest_lm.y;
        // use dists for gaussian weight calculation
        x_exp = std::pow(dist_x,2)/(2*std::pow(std_landmark[0],2));
        y_exp = std::pow(dist_y,2)/(2*std::pow(std_landmark[1],2));
        exponent = std::exp(-(x_exp + y_exp));
        // set weight according to distribution
        curr_weight = (pre_coeff * exponent);
        particles[particle_idx].weight *= curr_weight;
        weights[particle_idx] = particles[particle_idx].weight;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    std::default_random_engine gen;
	std::normal_distribution<> dx(0.0, std_landmark[0]);
	std::normal_distribution<> dy(0.0, std_landmark[1]);
    double lm_o_dist = 0;

	for(int i = 0; i < num_particles; i++){
	    std::vector<LandmarkObs> near_lms;
	    // find all landmarks within the laser range for that particle

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
            lm_o_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
            if (lm_o_dist < sensor_range){
                LandmarkObs lm;
                lm.x = map_landmarks.landmark_list[j].x_f + dx(gen);
                lm.y = map_landmarks.landmark_list[j].y_f + dy(gen);
                lm.id = map_landmarks.landmark_list[j].id_i;
                near_lms.push_back(lm);
            }
        }

        // map all observations to map frame
        std::vector<LandmarkObs> maped_observations;
        for(int k = 0; k < observations.size(); k++){
            LandmarkObs maped_ob;
            maped_ob.x = particles[i].x + std::cos(particles[i].theta)*observations[k].x - std::sin(particles[i].theta)*observations[k].y;
            maped_ob.y = particles[i].y + std::sin(particles[i].theta)*observations[k].x + std::cos(particles[i].theta)*observations[k].y;
            maped_observations.push_back(maped_ob);
        }

        dataAssociation(near_lms, maped_observations, std_landmark, i);
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine gen;
    std::discrete_distribution<int> index(weights.begin(), weights.end());

    std::vector<Particle> new_particles;

    for(int n=0; n<num_particles; n++) {
        int i = index(gen);

        Particle p;
        p.id = n;
        p.x = particles[i].x;
        p.y = particles[i].y;
        p.theta = particles[i].theta;
        p.weight = 1;

        new_particles.push_back(p);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
