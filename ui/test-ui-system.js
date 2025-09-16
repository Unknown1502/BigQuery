// Dynamic Pricing Intelligence UI System Test
// Comprehensive validation script for Node.js + React UI

const axios = require('axios');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5000';
const FRONTEND_URL = 'http://localhost:3000';

console.log('='.repeat(60));
console.log('Dynamic Pricing Intelligence UI System Test');
console.log('BigQuery AI Competition 2025');
console.log('='.repeat(60));

// Test configuration
const tests = [
  {
    name: 'Health Check',
    endpoint: '/api/health',
    method: 'GET'
  },
  {
    name: 'System Metrics',
    endpoint: '/api/metrics',
    method: 'GET'
  },
  {
    name: 'Real-time Pricing',
    endpoint: '/api/realtime-pricing',
    method: 'GET'
  },
  {
    name: 'Demand Forecasts',
    endpoint: '/api/forecasts',
    method: 'GET'
  },
  {
    name: 'Location Clusters',
    endpoint: '/api/clusters',
    method: 'GET'
  },
  {
    name: 'Feature Importance',
    endpoint: '/api/feature-importance',
    method: 'GET'
  },
  {
    name: 'Price Calculator',
    endpoint: '/api/calculate-price',
    method: 'POST',
    data: {
      latitude: 40.7128,
      longitude: -74.0060,
      hour: 14,
      day_of_week: 2,
      weather: 'clear',
      demand_level: 'high'
    }
  },
  {
    name: 'Performance Data',
    endpoint: '/api/performance',
    method: 'GET'
  }
];

// File structure validation
const requiredFiles = [
  'package.json',
  'server.js',
  'client/package.json',
  'client/src/App.js',
  'client/src/App.css',
  'client/src/index.js',
  'client/src/components/Dashboard.js',
  'client/src/components/RealTimePricing.js',
  'client/src/components/Forecasting.js',
  'client/src/components/LocationClusters.js',
  'client/src/components/FeatureImportance.js',
  'client/src/components/PriceCalculator.js',
  'client/src/components/SystemMetrics.js',
  'client/src/components/Header.js',
  'client/src/components/Sidebar.js',
  'client/src/components/LoadingSpinner.js',
  'client/public/index.html'
];

async function validateFileStructure() {
  console.log('\n📁 Validating File Structure...');
  
  let allFilesExist = true;
  
  for (const file of requiredFiles) {
    const filePath = path.join(__dirname, file);
    if (fs.existsSync(filePath)) {
      console.log(`✅ ${file}`);
    } else {
      console.log(`❌ ${file} - MISSING`);
      allFilesExist = false;
    }
  }
  
  return allFilesExist;
}

async function testAPIEndpoints() {
  console.log('\n🔌 Testing API Endpoints...');
  
  const results = [];
  
  for (const test of tests) {
    try {
      const config = {
        method: test.method,
        url: `${BASE_URL}${test.endpoint}`,
        timeout: 5000
      };
      
      if (test.data) {
        config.data = test.data;
      }
      
      const response = await axios(config);
      
      console.log(`✅ ${test.name}: ${response.status} ${response.statusText}`);
      results.push({
        test: test.name,
        status: 'PASS',
        statusCode: response.status,
        responseTime: response.headers['x-response-time'] || 'N/A'
      });
      
    } catch (error) {
      console.log(`❌ ${test.name}: ${error.message}`);
      results.push({
        test: test.name,
        status: 'FAIL',
        error: error.message
      });
    }
  }
  
  return results;
}

async function validatePackageFiles() {
  console.log('\n📦 Validating Package Configuration...');
  
  // Check backend package.json
  try {
    const backendPkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
    console.log(`✅ Backend package.json - ${backendPkg.name} v${backendPkg.version}`);
    
    const requiredDeps = ['express', '@google-cloud/bigquery', 'socket.io', 'cors', 'helmet'];
    for (const dep of requiredDeps) {
      if (backendPkg.dependencies && backendPkg.dependencies[dep]) {
        console.log(`  ✅ ${dep}: ${backendPkg.dependencies[dep]}`);
      } else {
        console.log(`  ❌ ${dep}: MISSING`);
      }
    }
  } catch (error) {
    console.log(`❌ Backend package.json: ${error.message}`);
  }
  
  // Check frontend package.json
  try {
    const frontendPkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'client/package.json'), 'utf8'));
    console.log(`✅ Frontend package.json - ${frontendPkg.name} v${frontendPkg.version}`);
    
    const requiredDeps = ['react', 'recharts', 'axios', 'socket.io-client'];
    for (const dep of requiredDeps) {
      if (frontendPkg.dependencies && frontendPkg.dependencies[dep]) {
        console.log(`  ✅ ${dep}: ${frontendPkg.dependencies[dep]}`);
      } else {
        console.log(`  ❌ ${dep}: MISSING`);
      }
    }
  } catch (error) {
    console.log(`❌ Frontend package.json: ${error.message}`);
  }
}

async function generateTestReport(fileValidation, apiResults) {
  console.log('\n📊 Test Summary Report');
  console.log('='.repeat(40));
  
  console.log(`File Structure: ${fileValidation ? 'PASS' : 'FAIL'}`);
  
  const passedTests = apiResults.filter(r => r.status === 'PASS').length;
  const totalTests = apiResults.length;
  
  console.log(`API Tests: ${passedTests}/${totalTests} PASSED`);
  
  if (passedTests === totalTests && fileValidation) {
    console.log('\n🎉 ALL TESTS PASSED - UI System Ready!');
    console.log(`Frontend: ${FRONTEND_URL}`);
    console.log(`Backend API: ${BASE_URL}`);
  } else {
    console.log('\n⚠️  Some tests failed - Check configuration');
  }
  
  console.log('\n🏆 BigQuery AI Competition 2025');
  console.log('Dynamic Pricing Intelligence UI Complete');
}

// Main test execution
async function runTests() {
  try {
    const fileValidation = await validateFileStructure();
    await validatePackageFiles();
    const apiResults = await testAPIEndpoints();
    await generateTestReport(fileValidation, apiResults);
  } catch (error) {
    console.error('Test execution failed:', error.message);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runTests();
}

module.exports = { runTests, validateFileStructure, testAPIEndpoints };
