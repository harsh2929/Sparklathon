/**
 * @type {import('next').NextConfig}
 */

// const withPWA = require('next-pwa');
// const runtimeCaching = require('next-pwa/cache');
const { i18n } = require('./next-i18next.config');

const moduleExports = {
  api: {
    bodyParser: {
      sizeLimit: '1mb'
    }
  },
  // compiler: {
  //   removeConsole: {
  //     exclude: ['error', 'warn']
  //   }
  // },
  async redirects() {
    return [
      {
        source: '/admin',
        destination: '/admin/dashboard',
        permanent: true
      }
    ];
  },
  i18n,
  // pwa: {
  //   disable: process.env.NODE_ENV === 'development',
  //   dest: 'public',
  //   runtimeCaching
  // },
  reactStrictMode: true,
  images: {
    deviceSizes: [320, 420, 768, 1024, 1200],
    // domains: ['127.0.0.1', 'digitaloceanspaces.com'],
    path: '/_next/image',
    loader: 'default',
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '*.digitaloceanspaces.com',
        port: '',
        pathname: '/**'
      }
    ]
  },
  env: {
    URL: 'https://myecomwebsite.vercel.app',
    GTAG_MEASUREMENT_ID: '',
    FB_APPID: '',
    // DATABASE
    POSTGRES_USER: 'don',
    POSTGRES_PASSWORD: '',
    POSTGRES_DB: 'production',
    PORT: 25060,
    DATABASE_END_POINT:
      
    // URL: 'http://localhost:3001',
    // POSTGRES_USER: 'crud_user',
    // POSTGRES_PASSWORD: 'crud_password',
    // POSTGRES_DB: 'development',
    // PORT: 5432,
    // DATABASE_END_POINT: '127.0.0.1',
    // S3 BUCKET
    S3_BUCKET_NAME: '',
    S3_REGION: 'fra1',
    S3_ACCESS_KEY_ID: ``,
    S3_SECRET_ACCESS_KEY:`',
    S3_ENDPOINT: ''
  },
  typescript: {
    ignoreBuildErrors: true
  },
  eslint: {
    ignoreDuringBuilds: true
  },
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production'
  }
};

module.exports = moduleExports;

// module.exports = withSentryConfig(
//   withPWA(moduleExports),
//   SentryWebpackPluginOptions
// );
