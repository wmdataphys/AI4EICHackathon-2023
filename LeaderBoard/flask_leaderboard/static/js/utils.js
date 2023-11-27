function calculateScrollbarWidth() {
    var scrollWidth = $('.tbl-content').width() - $('.tbl-content table').width();
    $('.tbl-header').css({ 'padding-right': scrollWidth });
  }
  
  // Attach the function to the window load and resize events
  $(window).on("load resize", calculateScrollbarWidth);