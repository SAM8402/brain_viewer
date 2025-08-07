# BFI Viewer

A Django-based brain imaging viewer application for displaying and processing brain tissue images, with support for multiple image formats and interactive visualization.

## Overview

BFI Viewer is a web application designed for viewing and analyzing brain tissue sections. It provides an interactive viewer with support for both JP2 (JPEG 2000) and BFI image formats, along with comprehensive image processing capabilities.

## Features

- **Interactive Brain Viewer**: Web-based viewer using OpenLayers for image visualization
- **Multi-format Support**: Handles JP2 and BFI image formats
- **Image Processing Pipeline**: Automated cropping, transformation, and enhancement tools
- **REST API**: RESTful endpoints for data access and manipulation
- **Dockerized Deployment**: Easy deployment using Docker containers
- **CORS Support**: Cross-origin resource sharing for frontend integration

## Project Structure

```
bfi_viewer/
├── app/
│   ├── backend/                 # Django backend application
│   │   ├── bfi_viewer/         # Main Django project
│   │   ├── brainviewer/        # Brain viewer app
│   │   │   ├── data/           # JSON transformation data
│   │   │   ├── static/         # Static files and images
│   │   │   ├── templates/      # HTML templates
│   │   │   └── views.py        # View controllers
│   │   └── manage.py           # Django management script
│   └── img_processing/         # Image processing scripts
│       └── Final_run_scripts/  # Production image processing tools
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose configuration
└── requirements.txt            # Python dependencies
```

## Prerequisites

- Python 3.8+
- Django 5.2+
- Docker (optional, for containerized deployment)

## Installation

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bfi_viewer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database**:
   ```bash
   cd app/backend
   python manage.py migrate
   ```

4. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

The application will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

The application will be available at `http://localhost:8000`

## Configuration

### Environment Variables

- `DJANGO_SETTINGS_MODULE`: Django settings module (default: `bfi_viewer.settings`)
- `DEBUG`: Debug mode (0 for production, 1 for development)
- `DOCKER_IP`: IP address for Docker port binding

### Settings

Key configuration options in `app/backend/bfi_viewer/settings.py`:

- **ALLOWED_HOSTS**: Configure allowed hostnames
- **CORS_ALLOW_ALL_ORIGINS**: CORS configuration for development
- **DATABASE**: SQLite database configuration
- **STATIC_URL**: Static file serving configuration

## API Endpoints

The application provides RESTful API endpoints for:

- Brain viewer data retrieval
- Image metadata access
- Section information queries

Authentication is handled via Django REST Framework with token authentication.

## Image Processing

The `img_processing` directory contains scripts for:

- **Image Cropping**: Automated content-aware cropping
- **Format Conversion**: Converting between image formats
- **Batch Processing**: Processing multiple images
- **Transformation**: Applying geometric transformations
- **Enhancement**: Image quality improvement

### Key Processing Scripts

- `crop_images.py`: Content-aware image cropping
- `batch_crop_images.py`: Batch processing for multiple images
- `convert_to_canvas.py`: Canvas format conversion
- `transform_find.ipynb`: Jupyter notebook for transformation analysis

## Usage

### Viewing Brain Sections

1. Navigate to the viewer interface
2. Select biosample ID (142, 222, or 244)
3. Choose section number
4. Use interactive controls to zoom and pan
5. Switch between JP2 and BFI views as needed

### Processing Images

Run image processing scripts from the `img_processing/Final_run_scripts/` directory:

```bash
python crop_images.py --input <input_path> --output <output_path>
```

## Development

### Adding New Features

1. Create feature branches from `dev`
2. Follow Django best practices for views and models
3. Add tests for new functionality
4. Update API documentation as needed

### Database Models

The application uses Django ORM with SQLite for data storage. Models are defined in `brainviewer/models.py`.

## Security Considerations

- The application includes CORS headers for cross-origin requests
- Token-based authentication for API access
- Docker deployment runs as non-root user
- Debug mode should be disabled in production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


## Support

For issues and questions, please refer to the project documentation or create an issue in the repository.
