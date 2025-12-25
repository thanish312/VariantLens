import { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  CssBaseline,
  ThemeProvider,
  Fade,
  SvgIcon,
} from '@mui/material';
import ShieldIcon from '@mui/icons-material/Shield';
import BiotechIcon from '@mui/icons-material/Biotech';
import axios from 'axios';

// Set the backend base URL for all axios requests
axios.defaults.baseURL = import.meta.env.VITE_API_URL || '';

import FileUpload from './components/FileUpload';
import ResultCard from './components/ResultCard';
import lightTheme from './theme/darkTheme'; // USE THE NEW LIGHT THEME

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a genome file (.vcf or .txt)');
      return;
    }
    setLoading(true);
    setResult(null);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(res.data);
    } catch (err) {
      setError(
        err?.response?.data?.error ||
          'Prediction failed. Please try again later.'
      );
    } finally {
      setLoading(false);
    }
  };

  const activeTheme = lightTheme; // Set the active theme

  return (
    <ThemeProvider theme={activeTheme}>
      <CssBaseline />
      {/* Decorative background icons */}
      <Box
        sx={{
          position: 'fixed',
          inset: 0,
          zIndex: 0,
          pointerEvents: 'none',
          overflow: 'hidden',
        }}
      >
        {/* Large Shield icon, top left */}
        <Box
          component={SvgIcon}
          sx={{
            position: 'absolute',
            top: { xs: -40, md: 20 },
            left: { xs: -40, md: 40 },
            fontSize: { xs: 180, md: 260 },
            color: (theme) =>
              theme.palette.mode === 'dark'
                ? 'rgba(255,255,255,0.04)'
                : 'rgba(0,0,0,0.04)',
            opacity: 1,
            filter: 'blur(0.5px)',
          }}
        >
          <ShieldIcon fontSize="inherit" />
        </Box>
        {/* Large DNA icon, bottom right */}
        <Box
          component={SvgIcon}
          sx={{
            position: 'absolute',
            bottom: { xs: -60, md: 0 },
            right: { xs: -60, md: 40 },
            fontSize: { xs: 200, md: 320 },
            color: (theme) =>
              theme.palette.mode === 'dark'
                ? 'rgba(255,255,255,0.06)'
                : 'rgba(0,0,0,0.06)',
            opacity: 1,
            filter: 'blur(0.5px)',
          }}
        >
          <BiotechIcon fontSize="inherit" />
        </Box>
      </Box>
      {/* Main content */}
      <Box // Main page wrapper
        sx={{
          minHeight: '100vh',
          width: '100vw',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'flex-start', // Align to top, let content flow
          py: { xs: 4, sm: 6, md: 8 },  // Generous vertical padding for the page
          px: { xs: 2, sm: 3 },        // Horizontal padding for the page
          overflowX: 'hidden',         // Prevent horizontal scroll
          position: 'relative',
          zIndex: 1,
          // background: activeTheme.palette.background.default, // Already handled by CssBaseline + body style in theme
        }}
      >
        <Container
          maxWidth="md" // <<--- INCREASED from "sm" for a wider content area
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center', // Center children horizontally
            textAlign: 'center',   // Default text alignment for children
            flexGrow: 1,         // Allow container to grow if content is short
            // minHeight removed, as main Box handles 100vh and flex-start pushes content down
            width: '100%',       // Ensure it uses available space up to maxWidth
          }}
        >
          <Box textAlign="center" mb={{xs: 5, sm: 7}} width="100%"> {/* Increased bottom margin */}
            <Typography
              variant="h1" // <<--- LARGER HEADING from theme
              component="h1" // Semantic main heading
              color="primary" // Use theme's primary color
              gutterBottom
              sx={{ fontWeight: 'fontWeightBold' }} // Use theme's bold variable
            >
              Gene Guard
            </Typography>
            <Typography
              variant="h3" // <<--- LARGER SUBTITLE from theme
              color="text.secondary" // Use theme's secondary text color
              sx={{
                maxWidth: '550px', // Constrain subtitle width slightly
                mx: 'auto',        // Center subtitle
                // fontWeight, letterSpacing, etc., from theme variant h5
              }}
            >
              AI-Driven Early Disease Prediction using Genomics
            </Typography>
          </Box>

          <Paper
            elevation={8}
            sx={{
              width: '100%',
              maxWidth: 600, // or any value you want
              mx: 'auto',
              p: { xs: 3, sm: 5 },
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              // Remove or adjust minHeight, height, etc. if present
            }}
          >
            <FileUpload
              file={file}
              loading={loading}
              error={error}
              onFileChange={handleFileChange}
              onSubmit={handleSubmit}
              // Pass size props for customization
              inputSize="large"
              buttonSize="large"
              textSize="h5"
            />
          </Paper>

          {result && (
            <Fade in={!!result} timeout={500}>
              <Box
                width="100%"
                maxWidth={{ xs: '95%', sm: 600, md: 750 }} // <<--- INCREASED MAX WIDTH for results
                mb={{xs: 6, sm: 8}} // Increased margin
              >
                <ResultCard result={result} />
              </Box>
            </Fade>
          )}

          <Box mt="auto" pt={{xs:6, sm:8}} pb={{xs:2, sm:3}} textAlign="center" width="100%"> {/* Pushes footer down, more padding */}
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ opacity: 0.75 }}
            >
              © {new Date().getFullYear()}{' '}
              <Typography component="span" variant="caption" sx={{ color: 'primary.main', fontWeight: 'fontWeightMedium' }}>
                Gene Guard
              </Typography>{' '}
              — Powered by Gemini AI
            </Typography>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
