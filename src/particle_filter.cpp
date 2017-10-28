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
#include <algorithm>

#include "particle_filter.h"

using namespace std;
void print_particles(const vector<Particle>& particles, const string& delimiter){
  cout<<delimiter<< endl;
  for(const auto& p : particles)
    p.print();
  cout<<delimiter<< endl;
}
void print_LandmarkObs(const vector<LandmarkObs>& obs, const string& delimiter){
  cout<<delimiter<< endl;
  for(const auto& p : obs)
    p.print();
  cout<<delimiter<< endl;
}
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  //Create a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  Particle p;
  num_particles = 100;
  for(int i = 0; i < num_particles; i++){
    p.id      = i;
    p.weight  = 1.0;
    p.x       = dist_x(gen);
    p.y       = dist_y(gen);
    p.theta   = dist_theta(gen);
    particles.push_back(p);
  }
  is_initialized=true;
  //print_particles(particles, "--- initialized particles ---");
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  vector<double> x_y_t;
  for(auto &p : particles){
    x_y_t = CalculatePrediction(p.x, p.y, velocity, p.theta, yaw_rate, delta_t);
    if(isnan(x_y_t[0]) || isnan(x_y_t[1]) || isnan(x_y_t[2])){
      cout << "WARNING prediction delivers nan values -> don't take them" << endl;
      p.print();
    }else{
      normal_distribution<double> dist_x(x_y_t[0], std_pos[0]);
      normal_distribution<double> dist_y(x_y_t[1], std_pos[1]);
      normal_distribution<double> dist_theta(x_y_t[2], std_pos[2]);
      p.x     = dist_x(gen);
      p.y     = dist_y(gen);
      p.theta = dist_theta(gen);
    }
  }
  //print_particles(particles, "--- predicted particles ---");
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  // more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  // according to the MAP'S coordinate system. You will need to transform between the two systems.
  // Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  // The following is a good resource for the theory:
  // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  // and the following is a good resource for the actual equation to implement (look at equation
  // 3.33
  // http://planning.cs.uiuc.edu/node99.html

  weights.clear(); //Clear previous weights

  //Calculate particle's weights for every associated land mark
  for(auto& p : particles){
    /*
     * In every cycle particle weight is
      calculated independent of the previous cycle
    */
    p.weight = 1.0;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    for(const auto& obs : observations){
      //Transform observation from vehicle coordinate systems to map coordinate system
      vector<double> TOBS = HomogenousTransform(p.x, p.y, obs.x, obs.y, p.theta);
      Map::single_landmark_s nearest_neighbor;
      double p_min_distance = 999999;
      //perform closest neighbor search
      for(const auto& lm : map_landmarks.landmark_list){
        //Consider only particle which are in sensor range
        if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range){
          double d = dist(TOBS[0], TOBS[1], lm.x_f, lm.y_f);
          //Take landmark with smallest distance
          if(d < p_min_distance){
            p_min_distance = d;
            nearest_neighbor = lm;
          }
        }
      }
      //Calculate weight for particle
      double cw = CalculatePWeight(TOBS[0], TOBS[1], nearest_neighbor.x_f, nearest_neighbor.y_f, std_landmark[0], std_landmark[1]);
      if(isnan(cw)){
        cout << "WARNING calculated weight is" << cw << endl;
        cout << "OBSR(" << obs.x << "," << obs.y << ")->"
             << "PART(" << p.x << "," << p.y << "," << p.theta << ")->"
             << "TOBS(" << TOBS[0] << "," << TOBS[1] << ")->"
             << "LANM(" << nearest_neighbor.x_f << "," << nearest_neighbor.y_f << "," << ")" << endl;
      }
      else{
        //Multiply weights for particle for all observations
        p.weight *= cw;
        p.associations.push_back(nearest_neighbor.id_i);
        p.sense_x.push_back(nearest_neighbor.x_f);
        p.sense_y.push_back(nearest_neighbor.y_f);
      }
    }
    weights.push_back(p.weight);
  }
  //cout << "weights " << weights.size() << " weights max " << *max_element(weights.begin(), weights.end())
  //        << " weight min " << *min_element(weights.begin(),weights.end()) << endl;
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.
  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution <> d(weights.begin(), weights.end());
  for(auto& p : particles){
    p = particles[d(gen)];
  }
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
