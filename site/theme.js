// Colour-scheme toggle for index.html.
//
// A separate same-origin file rather than an inline script element, so the
// page's Content-Security-Policy can say script-src 'self' instead of
// 'unsafe-inline', or a hash that stops matching the moment anyone edits a
// line here.
//
// It reads and writes exactly one localStorage key and one attribute on the
// root element. No network, no cookies, no user data, no third party. With
// JavaScript off the page is unchanged apart from the button doing nothing --
// the same two palettes come from prefers-color-scheme in gausscapture.css.
(function(){
  var btn=document.getElementById('themeBtn');
  var stored=null;
  try{stored=localStorage.getItem('gc-theme')}catch(e){}
  if(stored){document.documentElement.setAttribute('data-theme',stored)}
  btn.addEventListener('click',function(){
    var cur=document.documentElement.getAttribute('data-theme');
    if(!cur){
      cur=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
    }
    var next=cur==='dark'?'light':'dark';
    document.documentElement.setAttribute('data-theme',next);
    try{localStorage.setItem('gc-theme',next)}catch(e){}
  });
})();
