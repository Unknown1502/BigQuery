# Dynamic Pricing Intelligence UI/UX System

A comprehensive Node.js and React-based user interface for the Dynamic Pricing Intelligence System, built for the BigQuery AI Competition.

## 🏗️ Architecture

### Backend (Node.js + Express)
- **Express Server**: RESTful API with comprehensive endpoints
- **BigQuery Integration**: Direct connection to BigQuery ML models
- **WebSocket Support**: Real-time data streaming with Socket.io
- **Security**: CORS, Helmet, rate limiting, and input validation

### Frontend (React)
- **Modern React**: Hooks, functional components, and context
- **Responsive Design**: Tailwind CSS with mobile-first approach
- **Interactive Charts**: Recharts for data visualization
- **Real-time Updates**: Socket.io client for live data

## 🚀 Features

### Core Functionality
- **Dashboard**: System overview with key metrics and real-time charts
- **Real-time Pricing**: Live pricing data with WebSocket updates
- **Demand Forecasting**: ARIMA_PLUS model predictions with confidence intervals
- **Location Clustering**: K-MEANS clustering visualization and analysis
- **Feature Importance**: ML model feature analysis and insights
- **Price Calculator**: Interactive tool for optimal price calculation
- **System Metrics**: Performance monitoring and health status

### BigQuery ML Integration
- **ARIMA_PLUS**: Time series forecasting for demand prediction
- **K-MEANS**: Location clustering for market segmentation
- **BOOSTED_TREE**: Price optimization with feature importance
- **Real-time Queries**: Live data processing and model inference

## 📦 Installation

### Prerequisites
- Node.js 16+ and npm
- Google Cloud Project with BigQuery enabled
- Service account credentials for BigQuery access

### Backend Setup
```bash
cd ui
npm install
```

### Frontend Setup
```bash
cd ui/client
npm install
```

### Environment Configuration
Create `.env` file in the `ui` directory:
```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# BigQuery Configuration
BIGQUERY_DATASET_ID=ride_intelligence
BIGQUERY_LOCATION=US

# Server Configuration
PORT=5000
NODE_ENV=development

# CORS Configuration
CORS_ORIGIN=http://localhost:3000
```

## 🏃‍♂️ Running the Application

### Development Mode

1. **Start Backend Server**:
```bash
cd ui
npm run dev
```

2. **Start Frontend Development Server**:
```bash
cd ui/client
npm start
```

3. **Access Application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

### Production Mode

1. **Build Frontend**:
```bash
cd ui/client
npm run build
```

2. **Start Production Server**:
```bash
cd ui
npm start
```

## 🔌 API Endpoints

### Core Endpoints
- `GET /api/health` - System health check
- `GET /api/metrics` - System performance metrics
- `GET /api/realtime-pricing` - Real-time pricing data
- `GET /api/forecasts` - Demand forecasting predictions
- `GET /api/clusters` - Location clustering results
- `GET /api/feature-importance` - ML feature analysis
- `POST /api/calculate-price` - Price calculation
- `GET /api/performance` - Performance history

### WebSocket Events
- `connection` - Client connection established
- `pricing_update` - Real-time pricing updates
- `forecast_update` - New forecast data
- `system_alert` - System notifications

## 🎨 UI Components

### Layout Components
- **Header**: System status and navigation
- **Sidebar**: Feature navigation with icons
- **LoadingSpinner**: Consistent loading states

### Feature Components
- **Dashboard**: Overview with metrics and charts
- **RealTimePricing**: Live pricing with interactive charts
- **Forecasting**: ARIMA_PLUS predictions with confidence intervals
- **LocationClusters**: K-MEANS visualization and analysis
- **FeatureImportance**: ML model interpretability
- **PriceCalculator**: Interactive pricing tool
- **SystemMetrics**: Performance monitoring

## 📊 Data Visualization

### Chart Types
- **Line Charts**: Time series data and trends
- **Area Charts**: Confidence intervals and ranges
- **Bar Charts**: Categorical data and comparisons
- **Scatter Plots**: Clustering and correlation analysis
- **Pie Charts**: Distribution and proportions

### Interactive Features
- **Tooltips**: Detailed data on hover
- **Zoom/Pan**: Chart navigation
- **Filtering**: Data subset selection
- **Real-time Updates**: Live data streaming

## 🔧 Configuration

### Tailwind CSS Setup
The project uses Tailwind CSS for styling with custom configurations:
- Custom color palette for branding
- Responsive breakpoints
- Component utilities
- Animation classes

### Chart Configuration
Recharts is configured with:
- Responsive containers
- Custom color schemes
- Interactive tooltips
- Legend customization
- Animation effects

## 🧪 Testing

### Backend Testing
```bash
cd ui
npm test
```

### Frontend Testing
```bash
cd ui/client
npm test
```

### Integration Testing
```bash
# Test API endpoints
curl http://localhost:5000/api/health

# Test WebSocket connection
node test-websocket.js
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Cloud Deployment
1. **Google Cloud Run**:
```bash
gcloud run deploy pricing-ui --source .
```

2. **Kubernetes**:
```bash
kubectl apply -f k8s/
```

## 🔒 Security Features

### Backend Security
- **CORS**: Cross-origin request protection
- **Helmet**: Security headers
- **Rate Limiting**: API abuse prevention
- **Input Validation**: Request sanitization
- **Authentication**: Service account validation

### Frontend Security
- **Environment Variables**: Secure configuration
- **API Key Management**: Secure credential handling
- **XSS Protection**: Input sanitization
- **HTTPS Enforcement**: Secure communication

## 📈 Performance Optimization

### Backend Optimization
- **Connection Pooling**: BigQuery connection management
- **Caching**: Redis for frequent queries
- **Compression**: Gzip response compression
- **Async Processing**: Non-blocking operations

### Frontend Optimization
- **Code Splitting**: Lazy loading components
- **Memoization**: React.memo and useMemo
- **Bundle Optimization**: Webpack optimization
- **Image Optimization**: Responsive images

## 🐛 Troubleshooting

### Common Issues

1. **BigQuery Connection Error**:
   - Verify service account credentials
   - Check project ID and dataset configuration
   - Ensure BigQuery API is enabled

2. **WebSocket Connection Failed**:
   - Check CORS configuration
   - Verify server port accessibility
   - Review firewall settings

3. **Chart Rendering Issues**:
   - Verify data format compatibility
   - Check responsive container setup
   - Review console for JavaScript errors

### Debug Mode
Enable debug logging:
```env
DEBUG=true
LOG_LEVEL=debug
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

### Code Standards
- ESLint configuration for JavaScript
- Prettier for code formatting
- JSDoc for documentation
- Jest for testing

## 📄 License

This project is part of the BigQuery AI Competition submission and follows the competition guidelines and licensing terms.

## 🏆 Competition Features

### BigQuery ML Showcase
- **ARIMA_PLUS**: Advanced time series forecasting
- **K-MEANS**: Intelligent location clustering
- **BOOSTED_TREE**: Feature-rich price optimization
- **Real-time ML**: Live model inference and updates

### Innovation Highlights
- **Multimodal Intelligence**: Combining geospatial, temporal, and external data
- **Real-time Processing**: Live data streaming and analysis
- **Interactive Visualization**: Comprehensive data exploration
- **Production-Ready**: Scalable architecture and deployment

## 📞 Support

For technical support or questions about the Dynamic Pricing Intelligence UI system, please refer to the main project documentation or create an issue in the repository.

---

**Built for BigQuery AI Competition 2024**  
*Showcasing the power of BigQuery ML for real-world dynamic pricing intelligence*
