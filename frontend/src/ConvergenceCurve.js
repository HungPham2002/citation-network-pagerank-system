import React from 'react';
import { Line } from 'react-chartjs-2';
import './ConvergenceCurve.css';

function ConvergenceCurve({ convergenceData, algorithms }) {
  if (!convergenceData || convergenceData.length === 0) {
    return null;
  }

  const algorithmColors = {
    pagerank: '#0047AB',
    weighted_pagerank: '#FF6B35',
    hits: '#4CAF50'
  };

  const datasets = algorithms.map(algoId => {
    const algoData = convergenceData.find(d => d.algorithm === algoId);
    
    return {
      label: algoId.replace(/_/g, ' ').toUpperCase(),
      data: algoData?.residuals || [],
      borderColor: algorithmColors[algoId],
      backgroundColor: algorithmColors[algoId] + '20',
      borderWidth: 3,
      tension: 0.4,
      pointRadius: 4,
      pointHoverRadius: 6
    };
  });

  const chartData = {
    labels: convergenceData[0]?.residuals?.map((_, idx) => idx + 1) || [],
    datasets: datasets
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Algorithm Convergence Analysis',
        font: {
          size: 18,
          weight: 'bold'
        },
        color: '#0047AB'
      },
      legend: {
        display: true,
        position: 'top'
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Iteration',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      y: {
        type: 'logarithmic',
        title: {
          display: true,
          text: 'Residual (log scale)',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  return (
    <div className="convergence-curve-container">
      <div className="convergence-header">
        <h3>Convergence Curve Analysis</h3>
        <p>Lower residual = Better convergence. Logarithmic scale shows convergence speed.</p>
      </div>
      
      <div className="chart-wrapper">
        <Line data={chartData} options={chartOptions} />
      </div>

      <div className="convergence-insights">
        <h4>💡 Key Insights:</h4>
        <ul>
          {algorithms.map(algoId => {
            const algoData = convergenceData.find(d => d.algorithm === algoId);
            if (!algoData) return null;

            const iterations = algoData.residuals.length;
            const finalResidual = algoData.residuals[iterations - 1];

            return (
              <li key={algoId}>
                <strong style={{ color: algorithmColors[algoId] }}>
                  {algoId.replace(/_/g, ' ').toUpperCase()}:
                </strong> 
                {' '}Converged in {iterations} iterations with final residual {finalResidual?.toExponential(2)}
              </li>
            );
          })}
        </ul>

        <div className="convergence-explanation">
          <strong>Understanding Convergence:</strong>
          <p>
            The convergence curve shows how quickly each algorithm reaches a stable solution. 
            Steeper curves indicate faster convergence. The y-axis uses logarithmic scale to 
            better visualize small changes in residuals.
          </p>
        </div>
      </div>
    </div>
  );
}

export default ConvergenceCurve;