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
  num_particles = 1000;
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
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  vector<double> x_y_theta_pred;
  for(auto &p : particles){
    x_y_theta_pred = CalculatePrediction(p.x, p.y, velocity, p.theta, yaw_rate, delta_t);
    normal_distribution<double> dist_x(x_y_theta_pred[0], std_pos[0]);
    normal_distribution<double> dist_y(x_y_theta_pred[1], std_pos[1]);
    normal_distribution<double> dist_theta(x_y_theta_pred[2], std_pos[2]);
    p.x     = dist_x(gen);
    p.y     = dist_y(gen);
    p.theta = dist_theta(gen);
  }
  //print_particles(particles, "--- predicted particles ---");
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  print_LandmarkObs(observations, "--- Observations before association ---");
  for(auto& obs : observations){
    double min_dist = 1000; //initial value
    for(const auto& pre : predicted){
      vector <double> transformed_xy = HomogenousTransform(pre.x, pre.y, obs.x, obs.y);
      double d = dist(transformed_xy[0], transformed_xy[1], pre.x, pre.y);
      //cout << " min_dist " << min_dist << " distance btw predicted and observation " << d << endl;
      if(d < min_dist){
        obs.id = pre.id;
        obs.x = pre.x;
        obs.y = pre.y;
        min_dist = d;
      }
    }
  }
  //cout << "number of landmarks " << observations.size() << endl;
  print_LandmarkObs(observations, "--- Observations after association ---");
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
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

  /*
   * predict measurements to all the map landmarks
   * within sensor range for each particle.
   */
  vector<LandmarkObs> predicted;
  LandmarkObs obs;
  // TODO: don't add landmarks twice
  for(const auto& p : particles){
    for(const auto& lm : map_landmarks.landmark_list){
      if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range){
        obs.id  = lm.id_i;
        obs.x   = lm.x_f;
        obs.y   = lm.y_f;
        //avoid predicting the landmark multiple times
        if(find(predicted.begin(), predicted.end(), obs) == predicted.end())
          predicted.push_back(obs);
      }
    }
  }
  //print_LandmarkObs(predicted, "--- predicted mesurements within range ---");
  cout << predicted.size() << " predicted within range" << endl;
  vector<LandmarkObs> assLandMarks = observations;
  // Associate predicted observations which are closest to observed landmarks
  // NOTE: the assLandMarks will be converted to map coordinates after returning from dataAssociation
  dataAssociation(predicted, assLandMarks);
  weights.clear(); //Clear previous weights
  //Calculate particle's weights for every associated land mark
  for(auto& p : particles){
    for(const auto& lm : assLandMarks){
      for(const auto& obs : observations){
        //Multiply weights
        vector<double> tr_xy = HomogenousTransform(p.x, p.y, obs.x, obs.y, p.theta);
        double cw = CalculatePWeight(tr_xy[0], tr_xy[1], lm.x, lm.y, std_landmark[0], std_landmark[1]);
        p.weight *= cw;
        //if(cw!=0 or p.weight!=0)
          //cout << "cw: " << cw << " pw: " << p.weight << endl;
        //Normalization of weights, weights < 0 shouldn't be possible
        if(p.weight > 1){
          cout << "WARNING: weight " << p.weight << " > 1 -> set to 1" << endl;
          p.weight = 1;
        }
        //cout << "w: " << p.weight << endl;
      }
    }
    weights.push_back(p.weight);
  }
  cout << "predicted " << predicted.size() << " measurements" << endl;
  cout << "weights " << weights.size() << " weights max " << *max_element(weights.begin(), weights.end())
      << " weight min " << *min_element(weights.begin(),weights.end()) << endl;
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  //default_random_engine gen;
  //discrete_distribution <> d(weights.begin(), weights.end());
//  for(auto& p : particles){
//    p = particles[d(gen)];
//  }
  //cout << "random dd " << d(gen) << endl;




  random_device rd;  //Will be used to obtain a seed for the random number engine
  mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  uniform_int_distribution<> dis(0, num_particles);
  int index = dis(gen);
  double beta = 0;
  double wm = *max_element(weights.begin(), weights.end());
  vector<Particle> sampledParticles;
  for(int i=0; i < num_particles; i++){
      beta += dis(gen) * 2.0 * wm;
      while(beta > weights[index]){
          beta -= weights[index];
          index = (index + 1) % num_particles;
      }
      sampledParticles.push_back(particles[index]);
  }
  particles = sampledParticles;

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
