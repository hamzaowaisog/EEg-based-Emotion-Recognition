\documentclass{beamer}
\usetheme{metropolis}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{multicol}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{transparent}
% Set aspect ratio to 16:9
\usepackage{geometry}
\geometry{papersize={16cm,9cm}}

% Remove navigation symbols
\setbeamertemplate{navigation symbols}{}

% Colors
\definecolor{nuces}{RGB}{0, 51, 102}
\setbeamercolor{alerted text}{fg=nuces}
\setbeamercolor{frametitle}{bg=nuces,fg=white}

% Title page information
\title{Emotion Recognition Using Physiological Signals}
\subtitle{Final Year Project Presentation}

\author{Abdul Haseeb Dharwarwala (K21-3217) \\ Muhammad Hamza (K21-3815) \\ Aheed Tahir (K21-4517)\\ \\Supervisor: Dr. Kamran Ali\\}
\institute{Department of Computer Science \\ FAST-NUCES, Karachi Campus}
\date{\today}

\begin{document}

% Title slide
% Title slide
\begin{frame}[plain]
\titlepage
\vfill
\begin{center}
Supervisor: Dr. Kamran Ali\\
Co-Supervisor: Dr. Fahad Sherwani
\end{center}
\end{frame}

% Agenda slide
\begin{frame}
\frametitle{Agenda}
\begin{columns}[T]
\column{0.5\textwidth}
\begin{itemize}
\item FYP-1 Summary
\item Problem Statement
\item Research Objectives
\item Literature Review
\item Datasets
\end{itemize}

\column{0.5\textwidth}
\begin{itemize}
\item Approaches
  \begin{itemize}
  \item DEAP EEG + Face
  \item DEAP EEG-only
  \item SEED EEG-only
  \end{itemize}
\item Results \& Insights
\item Challenges
\item Future Work
\end{itemize}
\end{columns}
\end{frame}

% FYP-1 Summary slide (1 slide max)
\begin{frame}
\frametitle{FYP-1 Summary}

\begin{columns}[T]
\column{0.55\textwidth}
\textbf{Problem Solved:}
\begin{itemize}
\item Cross-subject EEG emotion recognition
\item Eliminating individual brain pattern differences
\item Creating subject-invariant representations
\end{itemize}

\textbf{Key Achievement:}
\begin{itemize}
\item 85.4\% accuracy on SEED dataset
\item CLISA framework implementation [1]
\end{itemize}

\column{0.45\textwidth}
\textbf{Limitations:}
\begin{itemize}
\item No explicit domain adaptation
\item Single modality only
\item Limited noise handling
\end{itemize}

\vspace{0.3cm}
\textbf{\textrightarrow{} FYP-2 Motivation:}\\
Extend with multi-source approach and multimodal fusion
\end{columns}
\end{frame}

% Problem Statement slide
\begin{frame}
\frametitle{Problem Statement}

\begin{alertblock}{Key Challenge}
How to develop EEG-based emotion recognition systems that generalize effectively across different subjects despite individual variability?
\end{alertblock}

\vspace{0.5cm}
\textbf{Specific Challenges:}
\begin{itemize}
\item \textbf{Inter-subject variability} in brain signals
\item \textbf{Multimodal integration} complexity
\item \textbf{Feature extraction} from noisy EEG
\item \textbf{Label noise \& class imbalance} in datasets
\end{itemize}

\end{frame}

% Objectives slide
\begin{frame}
\frametitle{Research Objectives}

\begin{columns}[T]
\column{0.6\textwidth}
\textbf{Primary Goals:}
\begin{enumerate}
\item Develop multimodal EEG+Face fusion system
\item Implement specialized EEG-only architecture
\item Design advanced loss functions for cross-subject learning
\item Compare approaches for real-world applicability
\end{enumerate}

\column{0.4\textwidth}
\textbf{Why It Matters:}
\begin{itemize}
\item Emotion-aware systems
\item Clinical applications
\item Human-computer interaction
\item Personalized experiences
\end{itemize}
\end{columns}

\end{frame}

% Literature Review slide
\begin{frame}
\frametitle{Literature Review}

\begin{table}
\centering
\begin{tabular}{p{0.45\textwidth} | p{0.45\textwidth}}
\textbf{CLISA [1] - FYP-1} & \textbf{MSCL [2] - FYP-2} \\
\hline
Contrastive learning for subject-invariant features & Multi-source contrastive learning approach \\
\hline
Single-stage representation learning & Dual-stage contrastive learning \\
\hline
Implicit domain alignment & Dynamic domain adaptation \\
\hline
Single modality (EEG only) & Designed for multi-source fusion \\
\hline
Limited noise handling & Prototype embeddings for noise robustness \\
\end{tabular}
\end{table}

\vspace{0.3cm}
\textbf{Key Techniques:}
\begin{itemize}
\item Contrastive learning for representation alignment
\item Maximum Mean Discrepancy (MMD) for distribution matching
\item Cross-modal contrastive alignment for multimodal fusion
\end{itemize}

\end{frame}

% Dataset Description slide
\begin{frame}
\frametitle{Dataset Description}
\small 
\begin{columns}[T,onlytextwidth]
% Left Column – Text
\column{0.55\textwidth}
{\footnotesize
\textbf{DEAP \cite{koelstra2011}:}
\begin{itemize}
  \setlength\itemsep{0.1em}
  \item 32 participants
  \item 40 one-minute music videos
  \item 32-channel EEG @ 128 Hz
  \item Facial video recordings
  \item Valence/arousal ratings (1–9)
  \item Used in Approaches 1 \& 2
\end{itemize}

\vspace{0.1cm}

\textbf{SEED [4]:}
\begin{itemize}
  \setlength\itemsep{0.1em}
  \item 15 participants
  \item Emotional film clips
  \item 62-channel EEG @ 200 Hz
  \item Three emotion classes: Positive, Neutral, Negative
  \item Three sessions per subject
  \item Used in Approach 3
\end{itemize}
}


% Right Column – Image
\column{0.45\textwidth}
\centering
\includegraphics[width=0.95\linewidth]{seedicon.png}
\vspace{0.2cm}

\small
\textit{Figure: Human brain channel representation}
\end{columns}

\end{frame}


% Approach 1 slide
\begin{frame}
\frametitle{Approach 1: DEAP EEG + Face Fusion}

\begin{columns}[T]
\column{0.5\textwidth}
\textbf{Key Components:}
\begin{itemize}
\item Binary valence classification
\item EEG + facial embeddings fusion
\item Cross-modal contrastive alignment
\item Subject-invariant representation
\end{itemize}

\column{0.5\textwidth}
\textbf{Training:}
\begin{itemize}
\item Combined loss function:
  \begin{itemize}
  \item Classification loss
  \item Supervised contrastive
  \item Cross-modal contrastive
  \end{itemize}
\end{itemize}
\end{columns}

% Figure at the bottom
\vspace{0.1cm}
\begin{figure}
\centering
\includegraphics[width=1.0\linewidth]{deaparc1.png}
\caption{EEG+Face Fusion Architecture}
\end{figure}

\end{frame}

% Approach 2 slide
\begin{frame}
\frametitle{Approach 2: DEAP EEG-Only (4-Class)}

% Text content at the top
\begin{columns}[T]
\column{0.5\textwidth}
\textbf{Architecture:}
\begin{itemize}
\item Common Feature Extractor
\item Subject-Specific Mapper
\item Cross-Subject Alignment
\item 4-class emotion prediction
\end{itemize}

\column{0.5\textwidth}
\textbf{Advanced Loss Functions:}
\begin{itemize}
\item Dynamic Weighted Focal Loss
\item Prototype Contrastive Loss
\item MMD Loss for domain alignment
\end{itemize}
\end{columns}

% Figure at the bottom
\vspace{0.1cm}
\begin{figure}
\centering
\includegraphics[width=0.9\linewidth]{deaparc2.png}
\caption{DEAP EEG-Only Architecture}
\end{figure}

\end{frame}

% Approach 3 slide
\begin{frame}
\frametitle{Approach 3: SEED EEG-Only (3-Class)}

% Text content at the top
\begin{columns}[T]
\column{0.5\textwidth}
\textbf{Pipeline:}
\begin{itemize}
\item DE features across 5 bands
\item DASM and RASM asymmetry
\item Deep encoder network
\item Generalized Cross-Entropy
\end{itemize}

\column{0.5\textwidth}
\textbf{Training Strategy:}
\begin{itemize}
\item Leave-One-Subject-Out CV
\item Class-balanced sampling
\item Early stopping (patience=15)
\end{itemize}
\end{columns}

% Figure at the bottom
\vspace{0.1cm}
\begin{figure}
\centering
\includegraphics[width=1.0\linewidth]{seedarc1.png}
\caption{SEED EEG-Only Architecture}
\end{figure}

\end{frame}

% Results slide 1
\begin{frame}
\frametitle{Experimental Results}

\begin{table}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{DEAP EEG+Face} & \textbf{DEAP EEG-only} & \textbf{SEED EEG-only} \\
\midrule
SVM + DE features & 49.1\% & 41.7\% & 65.2\% \\
CNN-based model & 52.3\% & 53.8\% & 71.8\% \\
CLISA (FYP-1) & N/A & N/A & 85.4\% \\
\textbf{Our approach} & \textbf{68.3\%} & \textbf{68.7\%} & \textbf{86.5\%} \\
\bottomrule
\end{tabular}
\caption{Subject-Independent Accuracy Comparison}
\end{table}

\vspace{0.3cm}
\begin{columns}[T]
\column{0.48\textwidth}
\textbf{DEAP EEG+Face Results:}
\begin{itemize}
\item Subject-dependent: 68.4\%
\item Subject-independent: 54.6\%
\item Significant cross-subject gap
\end{itemize}

\column{0.48\textwidth}
\textbf{SEED Ablation Study:}
\begin{itemize}
\item CE Loss only: 75.2\%
\item CE + MMD Loss: 80.9\%
\item CE + L\_con2 + MMD: \textbf{86.5\%}
\end{itemize}
\end{columns}

\end{frame}

% Results slide 2 - Visualizations
\begin{frame}
\frametitle{Feature Visualization}

\begin{figure}
\centering
\includegraphics[width=0.6\linewidth]{all_tsne.png}
\caption{t-SNE visualization showing improved class separation across approaches}
\end{figure}

\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\textbf{\% TODO: Add confusion matrix DEAP EEG+Face}
\end{center}

\column{0.48\textwidth}
\begin{center}
\textbf{\% TODO: Add confusion matrix SEED}
\end{center}
\end{columns}

\end{frame}

% Key Insights slide
\begin{frame}
\frametitle{Key Insights}

\begin{columns}[T]
\column{0.48\textwidth}
\textbf{What Worked Well:}
\begin{itemize}
\item Advanced loss functions outperformed standard approaches
\item Multi-source contrastive learning improved generalization
\item Cross-modal alignment enhanced multimodal fusion
\item Prototype-based learning reduced noise impact
\end{itemize}

\column{0.48\textwidth}
\textbf{Remaining Challenges:}
\begin{itemize}
\item Gap between subject-dependent and independent performance
\item EEG signal variability still limits cross-subject accuracy
\item Laboratory datasets may not represent real-world scenarios
\item Computational complexity for real-time applications
\end{itemize}
\end{columns}

\end{frame}

% Challenges + Future Work slide
\begin{frame}
\frametitle{Challenges \& Future Work}

% Image background (behind text)
\begin{tikzpicture}[remember picture,overlay]
  \node[opacity=0.2, anchor=center, xshift=-8cm, yshift=-6cm] at (current page.north east) {
    \includegraphics[width=1.0\paperwidth]{cf.png}
  };
\end{tikzpicture}

% Foreground content (text columns)
\begin{columns}[T]
\column{0.48\textwidth}
\textbf{Technical Challenges:}
\begin{itemize}
  \item High inter-subject variability
  \item Complex loss integration
  \item Hyperparameter sensitivity
  \item Feature normalization without data leakage
\end{itemize}

\column{0.48\textwidth}
\textbf{Future Directions:}
\begin{itemize}
  \item Real-time implementation optimization
  \item Transformer architectures for EEG
  \item Rapid personalization strategies
  \item Cross-dataset generalization
  \item Multimodal temporal alignment
\end{itemize}
\end{columns}
\end{frame}


% References slide
\begin{frame}[allowframebreaks]
\frametitle{References}

\begin{thebibliography}{5}
\setbeamertemplate{bibliography item}[text]

\bibitem{shen2022} 
X. Shen et al., ``Contrastive learning of subject-invariant EEG representations for cross-subject emotion recognition,'' \textit{IEEE Trans. Affective Comput.}, vol. 14, no. 3, pp. 2496--2511, 2022.

\bibitem{deng2024} 
X. Deng et al., ``A novel multi-source contrastive learning approach for robust cross-subject emotion recognition in EEG data,'' \textit{Biomed. Signal Process. Control}, vol. 97, p. 106716, 2024.

\bibitem{koelstra2011} 
S. Koelstra et al., ``DEAP: A database for emotion analysis using physiological signals,'' \textit{IEEE Trans. Affective Comput.}, vol. 3, no. 1, pp. 18-31, 2011.

\bibitem{zheng2015} 
W. L. Zheng and B. L. Lu, ``Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks,'' \textit{IEEE Trans. Autonomous Mental Development}, vol. 7, no. 3, pp. 162-175, 2015.

\bibitem{khosla2020} 
P. Khosla et al., ``Supervised contrastive learning,'' \textit{Advances in Neural Information Processing Systems}, vol. 33, pp. 18661-18673, 2020.
\end{thebibliography}

\end{frame}

% Thank You slide
\begin{frame}
\frametitle{Thank You}

\begin{center}
\large{Questions \& Comments Welcome}

\vspace{1.5cm}

\normalsize
Contact:\\
k213217@nu.edu.pk\\
k213815@nu.edu.pk\\
k214517@nu.edu.pk

\vspace{1cm}

Department of Computer Science\\
FAST-National University of Computer \& Emerging Sciences\\
Karachi Campus
\end{center}
\end{frame}

\end{document}