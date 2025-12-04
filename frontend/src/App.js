import React, { useState, useEffect } from 'react';
import './App.css';
import logo from './logo.webp';
import { Bar } from 'react-chartjs-2';
import ReactMarkdown from 'react-markdown';
import NetworkGraph from './NetworkGraph';
import NetworkMetrics from './NetworkMetrics';
import RoleModal from './RoleModal';
import RoleBadge from './RoleBadge';
import ExportDataButton from './ExportDataButton';
import AlgorithmSelector from './AlgorithmSelector';
import ComparisonView from './ComparisonView';
import FormulaDisplay from './FormulaDisplay';
import ConvergenceCurve from './ConvergenceCurve';
import ProgressBar from './ProgressBar';

import { 
  Chart as ChartJS, 
  BarElement, 
  CategoryScale, 
  LinearScale, 
  LogarithmicScale,
  Tooltip, 
  Legend,
  ArcElement,           
  RadialLinearScale,    
  PointElement,         
  LineElement,          
  Filler              
} from 'chart.js';

ChartJS.register(
  BarElement, 
  CategoryScale, 
  LinearScale, 
  LogarithmicScale,
  Tooltip, 
  Legend,
  ArcElement,           
  RadialLinearScale,    
  PointElement,         
  LineElement,          
  Filler               
);

const aboutContent = `# Citation Network PageRank System

This web application implements Google's original PageRank algorithm to analyze citation networks in academic research. Find the most influential papers and authors based on citation patterns.

## How to Use

### 👤 Author-Based Analysis
Enter a list of author names (one per line). The system will automatically collect their papers, build the citation network, and calculate PageRank scores to identify the most influential research papers.

### 📄 Paper-Based Analysis
Enter specific paper titles (one per line). The system will find these papers, analyze their citation relationships, and rank them by importance using PageRank algorithm.

## Input Examples

### Authors:
\`\`\`
Geoffrey Hinton
Yoshua Bengio
Yann LeCun
Andrew Ng
Fei-Fei Li
\`\`\`

### Papers:
\`\`\`
Attention Is All You Need
Deep Residual Learning for Image Recognition
ImageNet Classification with Deep Convolutional Neural Networks
BERT: Pre-training of Deep Bidirectional Transformers
Generative Adversarial Networks
\`\`\`

## Parameters

**Damping Factor (α)**: This parameter (default 0.85) represents the probability that a researcher continues following citations. Adjust this value (0.1 - 0.99) to simulate different citation following behaviors.

**Max Iterations**: The maximum number of iterations for the PageRank calculation. Default is 100; increase for larger networks or higher accuracy.

## Steps to Calculate Citation PageRank

1. Select input mode (Authors or Papers)
2. Enter author names or paper titles (one per line)
3. Adjust the Damping Factor and Max Iterations if needed
4. Click "Calculate Citation PageRank" to view results and visualization
5. Explore the citation network graph and metrics

## Stakeholders

**Researchers**: Find influential papers and authors in your field
**Data Scientists**: Analyze citation patterns and trends
**Academic Institutions**: Evaluate research impact and ranking
**Students**: Discover foundational papers in research areas

## About HCMUT

Ho Chi Minh City University of Technology (HCMUT) is one of the leading engineering and technology universities in Vietnam, under the Vietnam National University – Ho Chi Minh City (VNU-HCM). The university is renowned for its strong focus on innovation, research, and practical applications in science and technology, providing a high-quality education environment for both undergraduate and graduate students.

## About This Project

This project was developed as part of the Intelligent Systems (CO5119) course at Ho Chi Minh City University of Technology (HCMUT) – VNU-HCM, under the supervision of Assoc. Prof. Dr. Quan Thanh Tho.

## About Us
- **Phạm Hữu Hùng** — Postgraduate Student (ID: 2470299) • [CV (PDF)](https://github.com/HungPham2002/resume/blob/main/Resume_HungPham.pdf)
- **Võ Thị Vân Anh** — Postgraduate Student (ID: 2470283)

## Data Source

This system uses **Semantic Scholar API** which provides:
- 200M+ academic papers
- Citation relationships
- Author information
- Publication metadata

## Acknowledgment

The authors would like to express their sincere gratitude to BSc. Le Nho Han and BSc. Vu Tran Thanh Huong for their valuable suggestions and insightful reviews throughout the research and implementation of this project.
Semantic Scholar API for Providing metadata.
`;

function App() {
  // ===== BASIC STATES =====
  const [inputMode, setInputMode] = useState('authors');
  const [paperInputType, setPaperInputType] = useState('doi');
  const [authors, setAuthors] = useState('');
  const [papers, setPapers] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [page, setPage] = useState('home');
  const [networkData, setNetworkData] = useState(null);
  const [stats, setStats] = useState(null);
  const [networkMetrics, setNetworkMetrics] = useState(null);
  
  // ===== ROLE STATES =====
  const [userRole, setUserRole] = useState(null);
  const [permissions, setPermissions] = useState(null);
  const [isRoleModalOpen, setIsRoleModalOpen] = useState(false);

  // ===== ALGORITHM COMPARISON STATES =====
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(['pagerank']);
  const [algorithmParameters, setAlgorithmParameters] = useState({
    damping_factor: 0.85,
    max_iterations: 100
  });
  const [comparisonResults, setComparisonResults] = useState(null);
  const [singleAlgorithmResult, setSingleAlgorithmResult] = useState(null);
  const [showFormulaExplanation, setShowFormulaExplanation] = useState(false);

  // ===== SSE PROGRESS TRACKING STATES =====
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [progressStage, setProgressStage] = useState('init');
  const [isStreamingProgress, setIsStreamingProgress] = useState(false);

  // ===== EFFECTS =====
  useEffect(() => {
    const savedRole = localStorage.getItem('userRole');
    console.log('🔍 Checking saved role:', savedRole);
    if (savedRole) {
      setUserRole(savedRole);
      console.log('✅ Role loaded from localStorage:', savedRole);
    } else {
      console.log('❌ No role found, opening modal...');
      setIsRoleModalOpen(true);
    }
  }, []);

  useEffect(() => {
    if (userRole) {
      console.log('🔄 Role changed to:', userRole);
      // Clear tất cả results khi đổi role
      setResults([]);
      setComparisonResults(null);
      setSingleAlgorithmResult(null);
      setNetworkMetrics(null);
      setPermissions(null);
      
      // Reset algorithm selector về default cho Researcher
      if (userRole === 'researcher') {
        setSelectedAlgorithms(['pagerank']);
      }
    }
  }, [userRole]);  // Trigger khi userRole thay đổi

  // ===== HANDLER FUNCTIONS =====

  const handleSelectRole = (role) => {
    console.log('Role selected:', role);
    setUserRole(role);
    localStorage.setItem('userRole', role);
    
    // RESET STATE KHI CHỌN ROLE MỚI
    setResults([]);
    setComparisonResults(null);
    setSingleAlgorithmResult(null);
    setNetworkData(null);
    setStats(null);
    setNetworkMetrics(null);
    setPermissions(null);
    setSuccess('');
    setError('');
    setSelectedAlgorithms(['pagerank']);  // Reset về default
    
    setIsRoleModalOpen(false);
  };

  const handleChangeRole = () => {
    console.log('🔄 Opening role modal to change...');
    // RESET TẤT CẢ STATE KHI ĐỔI ROLE
    setResults([]);
    setComparisonResults(null);
    setSingleAlgorithmResult(null);
    setNetworkData(null);
    setStats(null);
    setNetworkMetrics(null);
    setPermissions(null);
    setSuccess('');
    setError('');
    setIsRoleModalOpen(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');
    setResults([]);
    setComparisonResults(null);
    setSingleAlgorithmResult(null);
    
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

    try {
      if (!userRole) {
        setError('Please select your role first');
        setLoading(false);
        return;
      }
      
      if (selectedAlgorithms.length > 1 && userRole === 'researcher') {
        setError('❌ Algorithm comparison is only available for Data Scientists. Please switch to Data Scientist role.');
        setLoading(false);
        return;
      }

      let endpoint = '';
      let requestBody = {
        damping_factor: algorithmParameters.damping_factor,
        max_iterations: algorithmParameters.max_iterations,
        user_role: userRole  // Gửi user role
      };

      // Xác định input mode
      if (inputMode === 'authors') {
        const authorList = authors.split('\n')
          .map(author => author.trim())
          .filter(author => author.length > 0);
        
        if (authorList.length === 0) {
          setError('Please enter at least one author name');
          setLoading(false);
          return;
        }
        
        requestBody.authors = authorList;
        requestBody.input_mode = 'authors';
      } else {
        const paperList = papers.split('\n')
          .map(paper => paper.trim())
          .filter(paper => paper.length > 0);
        
        if (paperList.length === 0) {
          setError(`Please enter at least one paper ${paperInputType === 'doi' ? 'DOI' : 'title'}`);
          setLoading(false);
          return;
        }
        
        requestBody.papers = paperList;
        requestBody.input_type = paperInputType;
        requestBody.input_mode = 'papers';
      }

      // LOGIC: Comparison mode hoặc Single algorithm
      if (selectedAlgorithms.length > 1) {
        endpoint = '/api/compare-algorithms';
        requestBody.algorithms = selectedAlgorithms;
        
        const response = await fetch(`${apiUrl}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        
        const data = await response.json();
        
        if (response.ok) {
          setComparisonResults(data);
          setNetworkData(data.network);
          setStats(data.stats);
          
          // CHỈ SET METRICS NẾU BACKEND TRẢ VỀ
          if (data.networkMetrics) {
            setNetworkMetrics(data.networkMetrics);
          } else {
            setNetworkMetrics(null);  // Clear metrics nếu không có quyền
          }
          
          // LẤY PERMISSIONS TỪ BACKEND
          setPermissions(data.permissions);
          setSuccess(`Successfully compared ${selectedAlgorithms.length} algorithms`);
        } else {
          setError(data.error || 'An error occurred');
        }
        
      } else {
        // SINGLE ALGORITHM MODE
        endpoint = '/api/calculate-with-algorithm';
        requestBody.algorithm = selectedAlgorithms[0];
        
        const response = await fetch(`${apiUrl}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        
        const data = await response.json();
        
        if (response.ok) {
          setSingleAlgorithmResult(data);
          setNetworkData(data.network);
          setStats(data.stats);
          
          // CHỈ SET METRICS NẾU BACKEND TRẢ VỀ
          if (data.networkMetrics) {
            setNetworkMetrics(data.networkMetrics);
          } else {
            setNetworkMetrics(null);  // Clear metrics
          }
          
          // Extract results
          if (data.algorithm === 'hits') {
            setResults(data.authority_results);
          } else {
            setResults(data.results);
          }
          
          // LẤY PERMISSIONS TỪ BACKEND
          setPermissions(data.permissions);
          setSuccess(`Successfully analyzed ${data.stats.totalPapers} papers`);
        } else {
          setError(data.error || 'An error occurred');
        }
      }
      
    } catch (err) {
      setError('Failed to connect to server: ' + err.message);
      console.error('Request error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ===== SSE HANDLER FUNCTION =====
  const handleCalculateWithProgress = async (e) => {
    e.preventDefault();
    setLoading(true);
    setIsStreamingProgress(true);
    setProgress(0);
    setProgressMessage('Starting...');
    setProgressStage('init');
    setError('');
    setSuccess('');
    setResults([]);
    setComparisonResults(null);
    setSingleAlgorithmResult(null);

    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

    try {
      // Validation
      if (!userRole) {
        setError('Please select your role first');
        setLoading(false);
        setIsStreamingProgress(false);
        return;
      }

      // ⭐ KIỂM TRA COMPARISON MODE
      if (selectedAlgorithms.length > 1 && userRole === 'researcher') {
        setError('❌ Algorithm comparison is only available for Data Scientists.');
        setLoading(false);
        setIsStreamingProgress(false);
        return;
      }

      // Prepare request body
      let requestBody = {
        damping_factor: algorithmParameters.damping_factor,
        max_iterations: algorithmParameters.max_iterations,
        user_role: userRole
      };

      // Xác định input mode
      if (inputMode === 'authors') {
        const authorList = authors.split('\n')
          .map(author => author.trim())
          .filter(author => author.length > 0);
        
        if (authorList.length === 0) {
          setError('Please enter at least one author name');
          setLoading(false);
          setIsStreamingProgress(false);
          return;
        }
        
        requestBody.authors = authorList;
        requestBody.input_mode = 'authors';
      } else {
        const paperList = papers.split('\n')
          .map(paper => paper.trim())
          .filter(paper => paper.length > 0);
        
        if (paperList.length === 0) {
          setError(`Please enter at least one paper ${paperInputType === 'doi' ? 'DOI' : 'title'}`);
          setLoading(false);
          setIsStreamingProgress(false);
          return;
        }
        
        requestBody.papers = paperList;
        requestBody.input_type = paperInputType;
        requestBody.input_mode = 'papers';
      }

      if (selectedAlgorithms.length > 1) {
        requestBody.algorithms = selectedAlgorithms;
        
        const response = await fetch(`${apiUrl}/api/compare-algorithms`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        
        const data = await response.json();
        
        if (response.ok) {
          setComparisonResults(data);
          setNetworkData(data.network);
          setStats(data.stats);
          
          if (data.networkMetrics) {
            setNetworkMetrics(data.networkMetrics);
          } else {
            setNetworkMetrics(null);
          }
          
          setPermissions(data.permissions);
          setSuccess(`Successfully compared ${selectedAlgorithms.length} algorithms`);
          setLoading(false);
          setIsStreamingProgress(false);
        } else {
          setError(data.error || 'An error occurred');
          setLoading(false);
          setIsStreamingProgress(false);
        }
        return; 
      }

      console.log('🚀 Starting SSE request...');

      const response = await fetch(`${apiUrl}/api/calculate-pagerank-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('✅ Stream complete');
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.progress !== undefined) {
                setProgress(data.progress);
              }
              
              if (data.message) {
                setProgressMessage(data.message);
              }
              
              if (data.stage) {
                setProgressStage(data.stage);
              }
              
              if (data.status === 'complete') {
                setResults(data.results || []);
                setNetworkData(data.network);
                setStats(data.stats);
                setPermissions(data.permissions);
                
                if (data.networkMetrics) {
                  setNetworkMetrics(data.networkMetrics);
                }
                
                setSingleAlgorithmResult({
                  results: data.results,
                  algorithm: 'pagerank',
                  performance: {
                    computation_time: 0,
                    iterations: 0
                  }
                });
                
                setSuccess(`Successfully analyzed ${data.stats?.totalPapers || 0} papers`);
                
                setTimeout(() => {
                  setLoading(false);
                  setIsStreamingProgress(false);
                }, 1000);
              }
              
              if (data.status === 'error') {
                console.error('❌ Stream error:', data.error);
                setError(data.error || 'An error occurred during calculation');
                setLoading(false);
                setIsStreamingProgress(false);
              }
              
            } catch (parseError) {
              console.error('❌ Error parsing SSE data:', parseError);
            }
          }
        }
      }
      
    } catch (error) {
      console.error('❌ Error in SSE connection:', error);
      setError(error.message || 'Failed to connect to server');
      setLoading(false);
      setIsStreamingProgress(false);
    }
  };

  // ===== CHART DATA =====
  const getScoreKey = () => {
    if (!selectedAlgorithms || selectedAlgorithms.length === 0) return 'pagerank';
    if (selectedAlgorithms[0] === 'pagerank') return 'pagerank';
    if (selectedAlgorithms[0] === 'weighted_pagerank') return 'weighted_pagerank';
    if (selectedAlgorithms[0] === 'hits') return 'authority_score';
    return 'pagerank';
  };

  const getChartTitle = () => {
    if (selectedAlgorithms[0] === 'pagerank') return 'PageRank Score Visualization';
    if (selectedAlgorithms[0] === 'weighted_pagerank') return 'Weighted PageRank Score Visualization';
    if (selectedAlgorithms[0] === 'hits') return 'HITS Authority Score Visualization';
    return 'Score Visualization';
  };

  const chartData = {
    labels: results.map(r => r.title.length > 40 ? r.title.substring(0, 40) + '...' : r.title),
    datasets: [
      {
        label: (selectedAlgorithms[0]?.replace(/_/g, ' ').toUpperCase() || 'PageRank') + ' Score',
        data: results.map(r => r[getScoreKey()] || 0),
        backgroundColor: 'rgba(0, 71, 171, 0.8)',
        borderColor: '#0047AB',
        borderWidth: 2,
        borderRadius: 10,
        borderSkipped: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { 
        display: false 
      },
      tooltip: { 
        enabled: true,
        backgroundColor: 'rgba(0, 71, 171, 0.95)',
        titleColor: '#FFD700',
        bodyColor: 'white',
        borderColor: '#FFD700',
        borderWidth: 2,
        cornerRadius: 10,
        displayColors: false,
        padding: 12,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        callbacks: {
          label: function(context) {
            return 'Score: ' + context.parsed.y.toFixed(6);
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { 
          color: '#0047AB',
          font: {
            weight: '600',
            size: 12
          }
        },
        grid: {
          color: 'rgba(0, 71, 171, 0.1)',
          drawBorder: false,
        },
        border: {
          color: 'rgba(0, 71, 171, 0.3)',
        }
      },
      x: {
        ticks: {
          color: '#0047AB',
          font: {
            weight: '600',
            size: 11
          },
          maxRotation: 45,
          minRotation: 45
        },
        grid: {
          color: 'rgba(0, 71, 171, 0.1)',
          drawBorder: false,
        },
        border: {
          color: 'rgba(0, 71, 171, 0.3)',
        }
      }
    }
  };

  return (
    <div className="App">
      <header className="hcmus-header">
        <div className="hcmus-header-content">
          <img src={logo} className="hcmus-logo" alt="HCMUT Logo" />
          <div>
            <h1 className="hcmus-title">Citation Network PageRank System</h1>
            <div className="hcmus-subtitle">Ho Chi Minh City University of Technology (HCMUT) - VNUHCM</div>
          </div>
          {userRole && (
            <RoleBadge role={userRole} onChangeRole={handleChangeRole} />
          )}
        </div>
      </header>
      
      <nav className="hcmus-navbar">
        <ul>
          <li className={page === 'home' ? 'active' : ''} onClick={() => setPage('home')}> 🏠︎ Home</li>
          <li className={page === 'about' ? 'active' : ''} onClick={() => setPage('about')}> 🕮 About</li>
          <li><a href="mailto:contact@hcmut.edu.vn" style={{ color: 'inherit', textDecoration: 'none' }}> ✉ Contact</a></li>
        </ul>
      </nav>
      
      <main className="App-main hcmus-main">
        {page === 'about' ? (
          <div className="input-section" style={{ maxWidth: 1000, margin: '0 auto', background: '#fafbff' }}>
            <ReactMarkdown>{aboutContent}</ReactMarkdown>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            {/* ALGORITHM SELECTOR CHO DATA SCIENTIST */}
            {userRole === 'data_scientist' && (
              <>
                <AlgorithmSelector 
                  onAlgorithmChange={setSelectedAlgorithms}
                  onParametersChange={setAlgorithmParameters}
                  selectedAlgorithms={selectedAlgorithms}
                  parameters={algorithmParameters}
                />
                
                {/* Formula Explanation Toggle */}
                <div style={{ textAlign: 'center', margin: '20px 0' }}>
                  <button
                    type="button"
                    className="btn-toggle-formula"
                    onClick={() => setShowFormulaExplanation(!showFormulaExplanation)}
                  >
                    {showFormulaExplanation ? 'Hide' : 'Show'} Algorithm Formulas & Explanation
                  </button>
                </div>

                {/* Formula Display */}
                {showFormulaExplanation && selectedAlgorithms.map(algo => (
                  <FormulaDisplay key={algo} algorithm={algo} />
                ))}
              </>
            )}

            <div className="input-section">
              <h2>🎓 Citation Network Analysis</h2>
              <p style={{ marginBottom: '20px', color: '#555', fontSize: '1.05em', lineHeight: '1.6' }}>
                Analyze citation networks by entering author names or paper titles to discover the most influential research.
              </p>

              {/* Input Mode Selection */}
              <label style={{ marginBottom: '10px', marginTop: '10px' }}>🔧 Input Mode:</label>
              <div style={{ marginBottom: '24px' }}>
                <label style={{ marginRight: '28px', cursor: 'pointer', fontSize: '1.05em' }}>
                  <input
                    type="radio"
                    name="inputMode"
                    value="authors"
                    checked={inputMode === 'authors'}
                    onChange={(e) => setInputMode(e.target.value)}
                    style={{ marginRight: '10px' }}
                  />
                  👤 Search by Authors
                </label>
                <label style={{ cursor: 'pointer', fontSize: '1.05em' }}>
                  <input
                    type="radio"
                    name="inputMode"
                    value="papers"
                    checked={inputMode === 'papers'}
                    onChange={(e) => setInputMode(e.target.value)}
                    style={{ marginRight: '10px' }}
                  />
                  📄 Search by Paper Titles
                </label>
              </div>

              {/* Conditional Input Fields */}
              {inputMode === 'authors' ? (
                <>
                  <label htmlFor="authors">👤 Author Names (one per line):</label>
                  <textarea
                    id="authors"
                    value={authors}
                    onChange={(e) => setAuthors(e.target.value)}
                    placeholder="Example:&#10;Tho Quan&#10;Geoffrey Hinton&#10;Yoshua Bengio&#10;Yann LeCun&#10;Andrew Ng"
                    rows="8"
                    style={{ 
                      fontSize: '1.05em',
                      lineHeight: '1.6'
                    }}
                  />
                  <p style={{ marginTop: '10px', fontSize: '14px', color: '#666', textAlign: 'left', background: '#f5f9ff', padding: '12px', borderRadius: '8px' }}>
                    💡 <strong>Author Mode:</strong> The system will find papers by these authors and analyze their citation networks.
                  </p>
                </>
              ) : (
                <>
                  {/* Sub-option: DOI or Title */}
                  <label style={{ marginBottom: '10px', marginTop: '10px' }}>Paper Input Type:</label>
                  <div style={{ marginBottom: '20px' }}>
                    <label style={{ marginRight: '28px', cursor: 'pointer', fontSize: '1.05em' }}>
                      <input
                        type="radio"
                        name="paperInputType"
                        value="doi"
                        checked={paperInputType === 'doi'}
                        onChange={(e) => setPaperInputType(e.target.value)}
                        style={{ marginRight: '10px' }}
                      />
                      DOI (Recommended - Fast & Accurate)
                    </label>
                    <label style={{ cursor: 'pointer', fontSize: '1.05em' }}>
                      <input
                        type="radio"
                        name="paperInputType"
                        value="title"
                        checked={paperInputType === 'title'}
                        onChange={(e) => setPaperInputType(e.target.value)}
                        style={{ marginRight: '10px' }}
                      />
                      Title (Slower - May be ambiguous)
                    </label>
                  </div>

                  {paperInputType === 'doi' ? (
                    <>
                      <label htmlFor="papers">Paper DOIs or arXiv IDs (one per line):</label>
                      <textarea
                        id="papers"
                        value={papers}
                        onChange={(e) => setPapers(e.target.value)}
                        placeholder="Example (multiple formats supported):&#10;10.1109/CVPR.2016.90&#10;arXiv:1706.03762&#10;1810.04805&#10;10.48550/arXiv.2010.11929&#10;2103.00020"
                        rows="8"
                        style={{ 
                          fontSize: '1.05em',
                          lineHeight: '1.6',
                          fontFamily: 'Consolas, Monaco, monospace'
                        }}
                      />
                      <p style={{ marginTop: '10px', fontSize: '14px', color: '#1b5e20', textAlign: 'left', background: '#e8f5e9', padding: '12px', borderRadius: '8px', borderLeft: '4px solid #4caf50' }}>
                        💡 <strong>Multiple Formats Supported:</strong>
                        <br /><br />
                        <strong>Standard DOI:</strong>
                        <br />
                        • ResNet: <code style={{ background: '#c8e6c9', padding: '2px 6px', borderRadius: '4px' }}>10.1109/CVPR.2016.90</code>
                        <br /><br />
                        <strong>arXiv Papers (3 formats work):</strong>
                        <br />
                        • Full DOI: <code style={{ background: '#c8e6c9', padding: '2px 6px', borderRadius: '4px' }}>10.48550/arXiv.1706.03762</code>
                        <br />
                        • With prefix: <code style={{ background: '#c8e6c9', padding: '2px 6px', borderRadius: '4px' }}>arXiv:1706.03762</code>
                        <br />
                        • Just ID: <code style={{ background: '#c8e6c9', padding: '2px 6px', borderRadius: '4px' }}>1706.03762</code>
                        <br /><br />
                        <strong>🔍 How to find:</strong> Google Scholar → Click paper → "Cite" → Look for DOI or arXiv ID
                      </p>
                    </>
                  ) : (
                    <>
                      <label htmlFor="papers">Paper Titles (one per line):</label>
                      <textarea
                        id="papers"
                        value={papers}
                        onChange={(e) => setPapers(e.target.value)}
                        placeholder="Example:&#10;Attention Is All You Need&#10;Deep Residual Learning for Image Recognition&#10;An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale&#10;BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
                        rows="8"
                        style={{ 
                          fontSize: '1.05em',
                          lineHeight: '1.6'
                        }}
                      />
                      <p style={{ marginTop: '10px', fontSize: '14px', color: '#e65100', textAlign: 'left', background: '#fff3e0', padding: '12px', borderRadius: '8px', borderLeft: '4px solid #ff9800' }}>
                        ⚠️ <strong>Title Mode:</strong> Searching by title is slower and may return multiple matches. Use DOI mode for better accuracy.
                      </p>
                    </>
                  )}
                </>
              )}

              {/* Advanced Parameters - CHỈ cho Data Scientist */}
              {userRole === 'data_scientist' && (
                <div style={{ marginTop: '28px', display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                  <div style={{ flex: '1', minWidth: '200px' }}>
                    <label htmlFor="damping">⚙️ Damping Factor (α):</label>
                    <input
                      id="damping"
                      type="number"
                      min="0.1"
                      max="0.99"
                      step="0.01"
                      value={algorithmParameters.damping_factor}
                      onChange={(e) => setAlgorithmParameters({
                        ...algorithmParameters,
                        damping_factor: parseFloat(e.target.value)
                      })}
                      style={{ 
                        width: '100%', 
                        padding: '10px', 
                        border: '2px solid #e8f0fe', 
                        borderRadius: '8px',
                        marginTop: '8px'
                      }}
                    />
                    <small style={{ display: 'block', color: '#666', marginTop: '6px' }}>
                      📌 Default: 0.85 (Google's standard)
                    </small>
                  </div>
                  
                  <div style={{ flex: '1', minWidth: '200px' }}>
                    <label htmlFor="iterations">🔄 Max Iterations:</label>
                    <input
                      id="iterations"
                      type="number"
                      min="10"
                      max="1000"
                      value={algorithmParameters.max_iterations}
                      onChange={(e) => setAlgorithmParameters({
                        ...algorithmParameters,
                        max_iterations: parseInt(e.target.value)
                      })}
                      style={{ 
                        width: '100%', 
                        padding: '10px', 
                        border: '2px solid #e8f0fe', 
                        borderRadius: '8px',
                        marginTop: '8px'
                      }}
                    />
                    <small style={{ display: 'block', color: '#666', marginTop: '6px' }}>
                      📌 Default: 100
                    </small>
                  </div>
                </div>
              )}

              {/* Info cho Researcher */}
              {userRole === 'researcher' && (
                <div style={{ marginTop: '20px', fontSize: '14px', color: '#555', textAlign: 'left', background: '#f0f7ff', padding: '14px', borderRadius: '8px', borderLeft: '4px solid #0047AB' }}>
                  ℹ️ <strong>Using default parameters:</strong> Damping Factor = 0.85, Max Iterations = 100
                  <br />
                  <small style={{ color: '#666', marginTop: '4px', display: 'block' }}>
                    (Advanced users can switch to Data Scientist role to customize these values)
                  </small>
                </div>
              )}

              <p style={{ marginTop: '20px', fontSize: '15px', color: '#555', textAlign: 'left', background: '#f5f9ff', padding: '16px', borderRadius: '8px', borderLeft: '4px solid #0047AB' }}>
                💡 <strong>How it works:</strong> The system fetches papers from Semantic Scholar, builds a citation network, and applies algorithms to identify the most influential research based on citation patterns.
              </p>
            </div>
            
            {/* Submit Button với 2 modes: Normal và SSE */}
            <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
              {/* Button 1: Normal mode (existing) */}
              {/* <button 
                type="submit" 
                disabled={loading}
                style={{ 
                  flex: '1', 
                  minWidth: '280px',
                  background: loading ? '#ccc' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                }}
              >
                {loading && !isStreamingProgress ? (
                  <>
                    <span className="loading"></span>
                    Analyzing Citations...
                  </>
                ) : (
                  '🚀 Calculate PageRank (Normal)'
                )}
              </button> */}

              {/* Button 2: SSE mode with progress */}
              <button 
                type="button"
                onClick={handleCalculateWithProgress}
                disabled={loading}
                style={{ 
                  flex: '1', 
                  minWidth: '280px',
                  background: loading ? '#ccc' : 'linear-gradient(135deg, #59abfeff 0%, #2857ffff 100%)'
                }}
              >
                {loading && isStreamingProgress ? (
                  <>
                    <span className="loading"></span>
                    Processing {progress}%...
                  </>
                ) : (
                  '🚀 Calculating PageRank 🚀'
                )}
              </button>
            </div>
          </form>
        )}

        {error && <div className="error">❌ {error}</div>}
        {success && <div className="success">{success}</div>}
        {/* Progress Bar - Hiển thị khi đang streaming */}
        {loading && isStreamingProgress && (
          <ProgressBar 
            progress={progress}
            message={progressMessage}
            stage={progressStage}
          />
        )}

        {/* COMPARISON RESULTS */}
        {comparisonResults && userRole === 'data_scientist' && (
          <>
            <ComparisonView 
              comparisonData={comparisonResults}
              algorithms={selectedAlgorithms}
            />

            {comparisonResults.convergence && 
            Array.isArray(comparisonResults.convergence) && 
            comparisonResults.convergence.length > 0 && (
              <ConvergenceCurve 
                convergenceData={comparisonResults.convergence}
                algorithms={selectedAlgorithms}
              />
            )}

            {/* Network Graph */}
            {networkData && (
              <NetworkGraph 
                results={comparisonResults.algorithms[selectedAlgorithms[0]]?.results || comparisonResults.algorithms[selectedAlgorithms[0]]?.authority_results || []}
                adjacencyMatrix={networkData.edges}
                urls={(comparisonResults.algorithms[selectedAlgorithms[0]]?.results || comparisonResults.algorithms[selectedAlgorithms[0]]?.authority_results || []).map(r => r.title.substring(0, 50))}
              />
            )}
          </>
        )}

        {/* SINGLE ALGORITHM RESULTS */}
        {singleAlgorithmResult && results.length > 0 && (
          <>
            {/* Export Button cho DS */}
            {permissions && permissions.export_data && (
              <div style={{ textAlign: 'center', marginBottom: '24px' }}>
                <ExportDataButton 
                  results={results}
                  network={networkData}
                  stats={stats}
                  networkMetrics={networkMetrics}
                  inputMode={inputMode}
                  parameters={algorithmParameters}
                  userRole={userRole}
                />
              </div>
            )}

            <div className="results">
              <h2>
                Top Influential Papers - {' '}
                {selectedAlgorithms[0] === 'pagerank' && 'PageRank'}
                {selectedAlgorithms[0] === 'weighted_pagerank' && 'Weighted PageRank'}
                {selectedAlgorithms[0] === 'hits' && 'HITS'}
              </h2>
              {stats && (
                <p style={{ marginBottom: '24px', color: '#555', textAlign: 'left', background: '#f5f9ff', padding: '16px', borderRadius: '8px' }}>
                  Analyzed <strong>{stats.totalPapers}</strong> papers with <strong>{stats.totalCitations}</strong> citation relationships.
                  <br />
                  <strong>Algorithm:</strong> {selectedAlgorithms[0].replace(/_/g, ' ').toUpperCase()}
                  <br />
                  <strong>Parameters:</strong> Damping Factor = {algorithmParameters.damping_factor}, Max Iterations = {algorithmParameters.max_iterations}
                  <br />
                  <strong>Performance:</strong> Computation Time = {singleAlgorithmResult.performance?.computation_time}s, 
                  Iterations = {singleAlgorithmResult.performance?.iterations}
                </p>
              )}

              {/* Results Table */}
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Paper Title</th>
                    <th>Authors</th>
                    <th>Year</th>
                    <th>Citations</th>
                    <th>
                      {selectedAlgorithms[0] === 'pagerank' && 'PageRank'}
                      {selectedAlgorithms[0] === 'weighted_pagerank' && 'Weighted PR'}
                      {selectedAlgorithms[0] === 'hits' && 'Authority Score'}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => {
                    const medalEmoji = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🏅';
                    
                    let score;
                    if (selectedAlgorithms[0] === 'pagerank') {
                      score = result.pagerank;
                    } else if (selectedAlgorithms[0] === 'weighted_pagerank') {
                      score = result.weighted_pagerank;
                    } else if (selectedAlgorithms[0] === 'hits') {
                      score = result.authority_score;
                    }

                    return (
                      <tr key={result.paperId}>
                        <td>
                          <strong style={{ color: '#0047AB', fontSize: '1.1em' }}>{medalEmoji} #{index + 1}</strong>
                        </td>
                        <td style={{ textAlign: 'left', maxWidth: '400px' }}>
                          <a 
                            href={`https://www.semanticscholar.org/paper/${result.paperId}`} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="App-link"
                          >
                            {result.title}
                          </a>
                        </td>
                        <td style={{ textAlign: 'left', maxWidth: '200px', fontSize: '0.95em' }}>
                          {result.authors.slice(0, 3).join(', ')}
                          {result.authors.length > 3 && ' et al.'}
                        </td>
                        <td>
                          <span style={{ fontWeight: '600' }}>{result.year || 'N/A'}</span>
                        </td>
                        <td>
                          <span style={{ color: '#2e7d32', fontWeight: '700' }}>{result.citationCount || 0}</span>
                        </td>
                        <td>
                          <strong style={{ color: '#0047AB', fontSize: '1.05em' }}>{score?.toFixed(6)}</strong>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Network Graph */}
            {networkData && (
              <NetworkGraph 
                results={results} 
                adjacencyMatrix={networkData.edges}
                urls={results.map(r => r.title.substring(0, 50))}
              />
            )}

            {/* Network Metrics - CHỈ cho Data Scientist */}
            {permissions && permissions.view_network_metrics && networkMetrics && (
              <NetworkMetrics 
                metrics={networkMetrics}
                results={results}
              />
            )}

            {/* Score Visualization Chart */}
            <div className="chart-container">
              <h3>{getChartTitle()}</h3>
              <div style={{ height: '450px', marginTop: '20px' }}>
                <Bar data={chartData} options={chartOptions} />
              </div>
            </div>
          </>
        )}
      </main>
      
      <footer className="hcmus-footer">
        <div style={{ marginBottom: '8px', fontSize: '1.05em', fontWeight: '600' }}>
          © {new Date().getFullYear()} Ho Chi Minh City University of Technology (HCMUT) | Citation Network PageRank System
        </div>
        <div>
          Contact: <a href="mailto:contact@hcmut.edu.vn">contact@hcmut.edu.vn</a> | 
          Website: <a href="https://www.hcmut.edu.vn" target="_blank" rel="noopener noreferrer">www.hcmut.edu.vn</a>
        </div>
      </footer>
      
      {/* Role Selection Modal */}
      <RoleModal 
        isOpen={isRoleModalOpen}
        onClose={() => setIsRoleModalOpen(false)}
        onSelectRole={handleSelectRole}
        currentRole={userRole}
      />
    </div>
  );
}

export default App;