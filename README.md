# CS-499
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ifeoluwa Adewoyin | Cybersecurity Professional Portfolio</title>
    <meta name="description" content="Computer Science graduate specializing in cybersecurity, showcasing enterprise-grade security implementations across mobile, web, and distributed systems.">
    
    <!-- Professional CSS Framework -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="assets/css/main.css" rel="stylesheet">
</head>
<body>
    <!-- Professional Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div class="container">
            <a class="navbar-brand fw-bold" href="#home">
                <i class="fas fa-shield-alt me-2"></i>Ifeoluwa Adewoyin
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link" href="#home">Home</a></li>
                    <li class="nav-item"><a class="nav-link" href="#about">About</a></li>
                    <li class="nav-item"><a class="nav-link" href="code-review/">Code Review</a></li>
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">
                            Enhancements
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="software-engineering/">
                                <i class="fas fa-mobile-alt me-2"></i>Software Engineering
                            </a></li>
                            <li><a class="dropdown-item" href="algorithms/">
                                <i class="fas fa-chart-line me-2"></i>Algorithms & Data Structures
                            </a></li>
                            <li><a class="dropdown-item" href="databases/">
                                <i class="fas fa-database me-2"></i>Database Systems
                            </a></li>
                        </ul>
                    </li>
                    <li class="nav-item"><a class="nav-link" href="contact/">Contact</a></li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section id="home" class="hero-section bg-gradient text-white py-5">
        <div class="container">
            <div class="row align-items-center min-vh-100">
                <div class="col-lg-8">
                    <h1 class="display-4 fw-bold mb-3">
                        Cybersecurity Professional Portfolio
                    </h1>
                    <p class="lead mb-4">
                        Computer Science graduate specializing in cybersecurity, with expertise in 
                        enterprise-grade security implementations across mobile, web, and distributed systems.
                    </p>
                    <div class="btn-group mb-4">
                        <a href="code-review/" class="btn btn-primary btn-lg">
                            <i class="fas fa-play me-2"></i>View Code Review
                        </a>
                        <a href="assets/documents/professional-self-assessment.pdf" class="btn btn-outline-light btn-lg">
                            <i class="fas fa-download me-2"></i>Download Resume
                        </a>
                    </div>
                    <div class="tech-stack">
                        <span class="badge bg-primary me-2 mb-2">Java/Android</span>
                        <span class="badge bg-primary me-2 mb-2">Python/Dash</span>
                        <span class="badge bg-primary me-2 mb-2">PostgreSQL</span>
                        <span class="badge bg-primary me-2 mb-2">MongoDB</span>
                        <span class="badge bg-primary me-2 mb-2">Cybersecurity</span>
                        <span class="badge bg-primary me-2 mb-2">Machine Learning</span>
                    </div>
                </div>
                <div class="col-lg-4 text-center">
                    <img src="assets/images/profile/professional-photo.jpg" 
                         alt="Ifeoluwa Adewoyin" 
                         class="img-fluid rounded-circle shadow-lg mb-3"
                         style="max-width: 300px;">
                </div>
            </div>
        </div>
    </section>

    <!-- Professional Self-Assessment Section -->
    <section id="about" class="py-5">
        <div class="container">
            <h2 class="text-center mb-5">Professional Self-Assessment</h2>
            <div class="row">
                <div class="col-lg-10 mx-auto">
                    <div class="professional-content">
                        <!-- Insert your professional self-assessment content here -->
                        <!-- Use clean typography and proper spacing -->
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Portfolio Overview -->
    <section class="py-5 bg-light">
        <div class="container">
            <h2 class="text-center mb-5">Portfolio Highlights</h2>
            <div class="row g-4">
                <div class="col-md-4">
                    <div class="card h-100 shadow">
                        <div class="card-body text-center">
                            <i class="fas fa-shield-alt fa-3x text-primary mb-3"></i>
                            <h5 class="card-title">Mobile Security</h5>
                            <p class="card-text">
                                Enterprise-grade Android security implementation with advanced 
                                authentication, encryption, and threat detection.
                            </p>
                            <a href="software-engineering/" class="btn btn-primary">Explore</a>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100 shadow">
                        <div class="card-body text-center">
                            <i class="fas fa-brain fa-3x text-success mb-3"></i>
                            <h5 class="card-title">Intelligent Algorithms</h5>
                            <p class="card-text">
                                Advanced algorithms including ML recommendations, fuzzy matching,
                                and performance optimization techniques.
                            </p>
                            <a href="algorithms/" class="btn btn-success">Explore</a>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100 shadow">
                        <div class="card-body text-center">
                            <i class="fas fa-server fa-3x text-info mb-3"></i>
                            <h5 class="card-title">Distributed Security</h5>
                            <p class="card-text">
                                Multi-database architecture with comprehensive security controls,
                                real-time monitoring, and scalable performance.
                            </p>
                            <a href="databases/" class="btn btn-info">Explore</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-dark text-white py-4">
        <div class="container text-center">
            <p>&copy; 2025 Ifeoluwa Adewoyin. CS 499 Computer Science Capstone Portfolio.</p>
            <div class="social-links">
                <a href="https://github.com/Ifeoluwa90" class="text-white me-3">
                    <i class="fab fa-github fa-lg"></i>
                </a>
                <a href="https://linkedin.com/in/your-profile" class="text-white me-3">
                    <i class="fab fa-linkedin fa-lg"></i>
                </a>
            </div>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script src="assets/js/portfolio.js"></script>
</body>
</html>
