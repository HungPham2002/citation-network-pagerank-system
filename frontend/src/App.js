import React, { useState } from 'react';
import './App.css';
import logo from './logo.webp';
import { Bar } from 'react-chartjs-2';
import ReactMarkdown from 'react-markdown';
import NetworkGraph from './NetworkGraph';
import NetworkMetrics from './NetworkMetrics';

import { 
  Chart as ChartJS, 
  BarElement, 
  CategoryScale, 
  LinearScale, 
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

### Author-Based Analysis
Enter a list of author names (one per line). The system will automatically collect their papers, build the citation network, and calculate PageRank scores to identify the most influential research papers.

### Parameters

**Damping Factor (α)**: This parameter (default 0.85) represents the probability that a researcher continues following citations. Adjust this value (0.1 - 0.99) to simulate different citation following behaviors.

**Max Iterations**: The maximum number of iterations for the PageRank calculation. Default is 100; increase for larger networks or higher accuracy.

## Steps to Calculate Citation PageRank

1. Enter author names (one per line)
2. Adjust the Damping Factor and Max Iterations if needed
3. Click "Calculate Citation PageRank" to view the results and visualization
4. Explore the citation network graph and metrics

## Stakeholders

🔬 **Researchers**: Find influential papers and authors in your field
📊 **Data Scientists**: Analyze citation patterns and trends
🏛️ **Academic Institutions**: Evaluate research impact and ranking

## About HCMUT

Ho Chi Minh City University of Technology (HCMUT) is one of the leading engineering and technology universities in Vietnam, under the Vietnam National University – Ho Chi Minh City (VNU-HCM). The university is committed to excellence in research and education.

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
`;

function App() {
  const [authors, setAuthors] = useState('');
  const [dampingFactor, setDampingFactor] = useState(0.85);
  const [maxIterations, setMaxIterations] = useState(100);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [page, setPage] = useState('home');
  const [networkData, setNetworkData] = useState(null);
  const [stats, setStats] = useState(null);
  const [networkMetrics, setNetworkMetrics] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');
    
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

    try {
      const authorList = authors.split('\n')
        .map(author => author.trim())
        .filter(author => author.length > 0);
      
      if (authorList.length === 0) {
        setError('Please enter at least one author name');
        setLoading(false);
        return;
      }
      
      const requestBody = {
        authors: authorList,
        damping_factor: dampingFactor,
        max_iterations: maxIterations
      };
      
      const response = await fetch(`${apiUrl}/api/calculate-citation-pagerank`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setResults(data.results);
        setNetworkData(data.network);
        setStats(data.stats);
        setNetworkMetrics(data.networkMetrics);
        setSuccess(`✅ Successfully analyzed ${data.stats.totalPapers} papers from ${authorList.length} authors`);
      } else {
        setError(data.error || 'An error occurred while analyzing citations');
      }
    } catch (err) {
      setError('Failed to connect to the server. Please make sure the backend is running on port 5000. Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = {
    labels: results.map(r => r.title.length > 40 ? r.title.substring(0, 40) + '...' : r.title),
    datasets: [
      {
        label: 'PageRank Score',
        data: results.map(r => r.pagerank),
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
            return 'PageRank: ' + context.parsed.y.toFixed(6);
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
            <div className="input-section">
              <h2>🎓 Citation Network Analysis</h2>
              <p style={{ marginBottom: '20px', color: '#555', fontSize: '1.05em', lineHeight: '1.6' }}>
                Enter author names to analyze their citation networks and discover the most influential research papers.
              </p>

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

              <div style={{ marginTop: '28px', display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                <div style={{ flex: '1', minWidth: '200px' }}>
                  <label htmlFor="damping">⚙️ Damping Factor (α):</label>
                  <input
                    id="damping"
                    type="number"
                    min="0.1"
                    max="0.99"
                    step="0.01"
                    value={dampingFactor}
                    onChange={(e) => setDampingFactor(parseFloat(e.target.value))}
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
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(parseInt(e.target.value))}
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

              <p style={{ marginTop: '20px', fontSize: '15px', color: '#555', textAlign: 'left', background: '#f5f9ff', padding: '16px', borderRadius: '8px', borderLeft: '4px solid #0047AB' }}>
                💡 <strong>How it works:</strong> The system fetches papers from Semantic Scholar, builds a citation network, and applies PageRank to identify the most influential research based on citation patterns.
              </p>
            </div>
            
            <button type="submit" disabled={loading}>
              {loading ? (
                <>
                  <span className="loading"></span>
                  Analyzing Citations...
                </>
              ) : (
                '🚀 Calculate Citation PageRank'
              )}
            </button>
          </form>
        )}

        {error && <div className="error">❌ {error}</div>}
        {success && <div className="success">{success}</div>}

        {results.length > 0 && (
          <>
            <div className="results">
              <h2>Top Influential Papers</h2>
              {stats && (
                <p style={{ marginBottom: '24px', color: '#555', textAlign: 'left', background: '#f5f9ff', padding: '16px', borderRadius: '8px' }}>
                  Analyzed <strong>{stats.totalPapers}</strong> papers with <strong>{stats.totalCitations}</strong> citation relationships.
                  <br />
                  <strong>Parameters used:</strong> Damping Factor (α) = {dampingFactor}, Max Iterations = {maxIterations}
                </p>
              )}
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Paper Title</th>
                    <th>Authors</th>
                    <th>Year</th>
                    <th>Citations</th>
                    <th>PageRank</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => {
                    const medalEmoji = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🏅';
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
                          <strong style={{ color: '#0047AB', fontSize: '1.05em' }}>{result.pagerank.toFixed(6)}</strong>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {networkData && (
              
                <NetworkGraph 
                  results={results} 
                  adjacencyMatrix={networkData.edges}
                  urls={results.map(r => r.title.substring(0, 50))}
                />
              
            )}

            {stats && results.length > 0 && (
              <NetworkMetrics 
                metrics={{
                  total_nodes: stats.totalPapers,
                  total_edges: stats.totalCitations,
                  density: stats.totalCitations / (stats.totalPapers * (stats.totalPapers - 1)),
                  avg_clustering_coefficient: 0.5, // Placeholder - backend cần tính
                  avg_in_degree: stats.totalCitations / stats.totalPapers,
                  avg_out_degree: stats.totalCitations / stats.totalPapers,
                  in_degree: results.map(r => r.citationCount || 0),
                  out_degree: results.map(r => 0), // Placeholder
                  strongly_connected_nodes: Math.floor(stats.totalPapers * 0.7),
                  dangling_nodes: Math.floor(stats.totalPapers * 0.1),
                  isolated_nodes: 0,
                  hubs: [],
                  authorities: results.slice(0, 5).map((r, idx) => ({
                    url: r.title,
                    in_degree: r.citationCount || 0,
                    score: (idx + 1) / 5
                  })),
                  hub_scores: results.slice(0, 5).map((r, idx) => 0.8 - (idx * 0.15)),
                  authority_scores: results.slice(0, 5).map((r, idx) => 0.9 - (idx * 0.15)),
                  degree_distribution: {}
                }}
                results={results}
              />
            )}

            <div className="chart-container">
              <h3>PageRank Score Visualization</h3>
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
    </div>
  );
}

export default App;