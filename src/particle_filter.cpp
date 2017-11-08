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
        weights.push_back(1);
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
	std::cout << "delta_t : " << delta_t << "std_pos: " << std_pos << " vel: " << velocity << "yaw_rate : " << yaw_rate << std::endl;
	for(int i = 0; i < num_particles; i++){
        const double theta = particles[i].theta;
        const double next_theta = theta + yaw_rate*delta_t;

        if (std::abs(yaw_rate) < 0.0001){
            // straight motion
            double vel = velocity * delta_t;
            particles[i].x += vel * std::sin(theta);
            particles[i].y += vel * std::cos(theta);
        }else{
            //turning motion
            double vel = velocity/yaw_rate;
            particles[i].x += vel * (std::sin(next_theta) - sin(theta));
            particles[i].y += vel * (std::cos(theta) - cos(next_theta));
            particles[i].theta += yaw_rate*delta_t;
        }

        //ad noise
        std::normal_distribution<> dx(0.0, std_pos[0]);
        particles[i].x += dx(gen);
        std::normal_distribution<> dy(0.0, std_pos[1]);
        particles[i].y += dy(gen);
        std::normal_distribution<> dyaw(0.0, std_pos[2]);
        particles[i].theta += dyaw(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    for(int i = 0; i < predicted.size(); i++){
        //TODO better use double max value here
        double min_dist = 1000000;
        for(int j = 0; j < observations.size(); j++){
            //find min distance
            double distance = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
            if (distance < min_dist){
                observations[j].id = predicted[i].id;
                min_dist = distance;
            }
        }
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


	for(int i = 0; i < num_particles; i++){
        Particle p = particles[i];
	    std::vector<LandmarkObs> predicted;
	    // find all landmarks within the laser range for that particle
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
            double lm_o_dist = dist(p.x, p.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
            if (lm_o_dist < sensor_range){
                LandmarkObs lm;
                lm.x = map_landmarks.landmark_list[j].x_f;
                lm.y = map_landmarks.landmark_list[j].y_f;
                lm.id = map_landmarks.landmark_list[j].id_i;
                predicted.push_back(lm);
            }
        }

        // map all observations to map frame
        std::vector<LandmarkObs> maped_observations;
        for(int k = 0; k < observations.size(); k++){
            LandmarkObs maped_ob;
            maped_ob.x = p.x + std::cos(p.theta)*observations[k].x - std::sin(p.theta)*observations[k].y;
            maped_ob.y = p.y + std::sin(p.theta)*observations[k].x + std::cos(p.theta)*observations[k].y;
            maped_observations.push_back(maped_ob);
        }

        dataAssociation(predicted, maped_observations);

        // iterate through all observations and calc min dist to landmark
        for(int j = 0; j < maped_observations.size(); j++){
            for (int k = 0; k < predicted.size(); k++){
                if (maped_observations[j].id == predicted[k].id){
                    double dist_x = maped_observations[j].x - predicted[k].x;
                    double dist_y = maped_observations[j].y - predicted[k].y;
                    double pre_coeff = 1./(2*M_PI*std_landmark[0]*std_landmark[1]);
                    // use dists for gaussian weight calculation
                    double x_exp = std::pow(dist_x,2)/(2*std::pow(std_landmark[0],2));
                    double y_exp = std::pow(dist_y,2)/(2*std::pow(std_landmark[1],2));
                    double exponent = std::exp(-(x_exp + y_exp));
                    // set weight according to distribution
                    p.weight = pre_coeff * exponent;
                }
            }
        }

        //normalize the weights for later usage as prob dist.
        p.weight /= num_particles;
        weights[i] = p.weight;
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
        const int i = index(gen);

        Particle p;
        p.id = i;
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
