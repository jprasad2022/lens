#!/usr/bin/env node
/**
 * Test script to verify frontend Kafka integration
 * Run this to test the API endpoints that the frontend will call
 */

const axios = require('axios');

const API_BASE = 'http://localhost:8000';
const TEST_USER_ID = 1;
const TEST_MOVIE_ID = 1;

async function testRatingEndpoint() {
  console.log('\n1. Testing Rating Endpoint:');
  try {
    const response = await axios.post(`${API_BASE}/users/${TEST_USER_ID}/rate`, {
      movie_id: TEST_MOVIE_ID,
      rating: 4.5
    });
    console.log('   ✅ Success:', response.data);
  } catch (error) {
    console.log('   ❌ Error:', error.response?.data || error.message);
  }
}

async function testWatchEndpoint() {
  console.log('\n2. Testing Watch Endpoint:');
  try {
    const response = await axios.post(`${API_BASE}/users/${TEST_USER_ID}/watch`, {
      movie_id: TEST_MOVIE_ID,
      progress: 1.0
    });
    console.log('   ✅ Success:', response.data);
  } catch (error) {
    console.log('   ❌ Error:', error.response?.data || error.message);
  }
}

async function testRecommendationTracking() {
  console.log('\n3. Testing Recommendation Tracking:');
  try {
    const response = await axios.get(`${API_BASE}/recommend/${TEST_USER_ID}?k=5`);
    console.log('   ✅ Success: Got', response.data.recommendations.length, 'recommendations');
    console.log('   Request ID:', response.data.request_id);
  } catch (error) {
    console.log('   ❌ Error:', error.response?.data || error.message);
  }
}

async function main() {
  console.log('=== Testing Frontend Kafka Integration ===');
  
  await testRatingEndpoint();
  await testWatchEndpoint();
  await testRecommendationTracking();
  
  console.log('\n=== Frontend Integration Summary ===');
  console.log('1. Rating endpoint: /users/{userId}/rate');
  console.log('2. Watch endpoint: /users/{userId}/watch');
  console.log('3. Recommendation tracking: Automatic on /recommend/{userId}');
  console.log('\nAll endpoints should produce events to Kafka topics.');
}

// Check if axios is installed
try {
  require.resolve('axios');
  main();
} catch(e) {
  console.log('Please install axios first: npm install axios');
}